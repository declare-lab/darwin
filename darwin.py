import torch.nn.functional as F
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import json
import numpy as np
from typing import List
from random import sample
from collections import Counter
from utils import Archive, apply_mistral_instruct_template, apply_llama3_template


class BestOfN:

    def __init__(self, model_name: str,
                 reward_model_name: str,
                 device="cuda"):
        if 'Llama-3' in model_name:  # Somehow loading the llama3 tokenizer from simpo checkpoints causes error, hence we load the default llama3 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code=True)
        elif 'Mistral' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'mistralai/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map=device, torch_dtype=torch.float16)
        self.device = device
        self.model_name = model_name
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_name)
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=reward_model_name,
            device=device,
            tokenizer=self.reward_tokenizer,
            torch_dtype=torch.float16
        )

    @torch.no_grad()
    def compute_sequence_score(self, query: str, generated_response: str):

        # Reward should be measured against seed instruction
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1
        }
        generated_response = generated_response.strip(self.tokenizer.eos_token)
        chat = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": generated_response}
        ]
        texts = [self.reward_tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False).replace(self.reward_tokenizer.bos_token, "")]
        pipe_outputs = self.rm_pipe(texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards[0]

    @torch.no_grad()
    def generate(self, query: str, n=5):
        if "Mistral" in self.model_name:
            query = apply_mistral_instruct_template(query, self.tokenizer)
        if 'Llama-3' in self.model_name:
            query = apply_llama3_template(query, self.tokenizer)

        inputs = self.tokenizer(query, return_tensors='pt')['input_ids']
        input_len = len(inputs[0])  # length of seed instruction

        input_ids = inputs.to(self.device)

        output_ids = self.model.generate(input_ids=input_ids,
                                         max_new_tokens=2048,
                                         do_sample=True,
                                         top_k=40,
                                         temperature=0.7,
                                         num_return_sequences=n).cpu()

        generated_strings = [self.tokenizer.decode(
            output_ids[i][input_len:], skip_special_tokens=True) for i in range(n)]

        reward_list = [self.compute_sequence_score(
            query, generated_string) for generated_string in generated_strings]

        index = np.argmax(reward_list)
        best_output = generated_strings[index].strip(self.tokenizer.eos_token)

        return best_output


class Darwin:
    def __init__(self,
                 model_name: str,
                 reward_model_name: str,
                 archive=None,
                 device="cuda"):
        if 'Llama-3' in model_name:
            # Somehow loading the llama3 tokenizer from simpo/dpo checkpoints causes error, hence we load the default llama3 tokenizer
            # If you are using llama3 models that add tokens to the tokenizer, please change this line accordingly
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code=True, padding_side='left')
        elif 'Mistral' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'mistralai/Mistral-7B-Instruct-v0.2', trust_remote_code=True, padding_side='left')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True,
                                                          device_map=device,
                                                          torch_dtype=torch.float16)
        # self.tokenizer.padding_side = 'left'
        if not self.tokenizer.pad_token:
            if 'Llama-3' in model_name:
                # For llama3 use the <|end_of_text|> as pad token which is 128001
                self.tokenizer.pad_token_id = 128001
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("No default pad token is set!")
                print("Using {} as pad token!".format(self.tokenizer.pad_token))
        self.device = device
        self.model_name = model_name
        self.archive = archive
        self.winning_beam_data = []
        self.winning_beam = None
        self.lc_step = 5

        self.winning_beam_replacement_history = []
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_name)
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=reward_model_name,
            device=device,
            tokenizer=self.reward_tokenizer,
            torch_dtype=torch.float16
        )

    def _process_mutation(self, generated_string: str, previous_instruction: str):

        lines = [line.strip()
                 for line in generated_string.strip().split('\n') if line]

        # Remove the numbering and store the instructions
        try:
            lines = [line for line in lines if line[0].isdigit()]
            instructions = [line.split('. ', 1)[1].strip(
                self.tokenizer.eos_token) for line in lines if line]
            if len(instructions) == 5:
                return instructions
            return [previous_instruction]*5
        except:
            print("Mutation failed!")
        return [previous_instruction]*5

    @torch.inference_mode
    def mutate(self, previous_instruction: str, n: int = 1):
        template = f''' You are a professional prompt engineer. You are given an original instruction and your goal is to mutate the instruction into 5 different instruction that will improve the clarity of original instruction. The mutated instruction should not deviate from the original instruction and they should provide the same general intention.
        Hint: Think of adding more details,removing details in the instruction or change certain phrasing when mutating the instruction.
        Only give the mutated instruction in a list order.
        Original instruction: How to make a cake?
        1. How to bake a delicious cake?
        2. Step-by-step guide to making a perfect cake from scratch
        3. How to bake a cake?
        4. Detailed instructions for creating a professional-quality cake at home
        5. How to prepare a beautiful homemade cake?
        Original instruction: {previous_instruction}?
        '''

        if "Mistral" in self.model_name:
            query = apply_mistral_instruct_template(template, self.tokenizer)
        if 'Llama-3' in self.model_name:
            query = apply_llama3_template(template, self.tokenizer)

        inputs = self.tokenizer(query, return_tensors='pt')['input_ids']
        input_len = len(inputs[0])  # length of seed instruction

        input_ids = inputs.to(self.device)

        output_ids = self.model.generate(input_ids=input_ids,
                                         max_new_tokens=1024,
                                         do_sample=True,
                                         top_k=40, temperature=0.7,
                                         num_return_sequences=n).cpu()

        generated_strings = [self.tokenizer.decode(
            output_ids[i][input_len:], skip_special_tokens=True) for i in range(n)]
        # Remove the numbering and store the instructions
        out = []
        for generated_string in generated_strings:
            out += self._process_mutation(generated_string,
                                          previous_instruction)
        assert (len(out) == 5*n)
        return out

    @torch.inference_mode
    def compute_sequence_score_lookahead(self,
                                         augmented_instructions: List[str],
                                         previous_states: List[str],
                                         state_complete: List[bool],
                                         lookahead: int
                                         ):
        candidate_states = []
        inp_len = []
        for i in range(len(state_complete)):
            if not state_complete[i]:
                # Continue exploring from previous state
                candidate_states.append(
                    augmented_instructions[i]+previous_states[i])
                inp_len.append(
                    len(self.tokenizer.encode(augmented_instructions[i])))

        inputs = self.tokenizer(
            candidate_states, padding=True, return_tensors='pt')
        # Compute how many pad tokens each sequence is added
        pad_length = torch.sum(inputs['attention_mask'] == 0, axis=1)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(**inputs,
                                  max_new_tokens=lookahead,
                                  do_sample=True,
                                  temperature=0.7,
                                  top_k=40)

        j = 0
        new_state = []
        for i in range(len(previous_states)):
            if not state_complete[i]:
                output_ids = out[j, inp_len[j]+pad_length[j]:]
                generated_text = self.tokenizer.decode(
                    output_ids, skip_special_tokens=False)
                # previous_states[i] = generated_text
                new_state.append(generated_text)
                j += 1
            else:
                new_state.append(previous_states[i])
        return self.compute_sequence_score(new_state)

    @torch.inference_mode
    def compute_sequence_score(self,
                               generated_response: List[str],
                               ):

        # Reward should be measured against seed instruction
        texts = []
        if not isinstance(generated_response, list):
            print("Input to reward model should be a list of strings!")
            return
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": len(generated_response)
        }
        for response in generated_response:

            response = self.tokenizer.decode(
                self.tokenizer.encode(response), skip_special_tokens=True)
            chat = [
                {"role": "user", "content": self.archive.seed_instruction},
                {"role": "assistant", "content": response}
            ]
            texts.append(self.reward_tokenizer.apply_chat_template(chat, tokenize=False,
                                                                   add_generation_prompt=False).replace(self.reward_tokenizer.bos_token, ""))

        pipe_outputs = self.rm_pipe(texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]

        return rewards

    @torch.inference_mode
    def compute_original_instruction_score(self):
        if "Mistral" in self.model_name:
            query = apply_mistral_instruct_template(
                self.archive.seed_instruction, self.tokenizer)
        if 'Llama-3' in self.model_name:
            query = apply_llama3_template(
                self.archive.seed_instruction, self.tokenizer)

        inputs = self.tokenizer(query, return_tensors='pt')['input_ids']
        input_len = len(inputs[0])  # length of seed instruction

        input_ids = inputs.to(self.device)

        output_ids = self.model.generate(input_ids=input_ids,
                                         max_new_tokens=2048,
                                         do_sample=True,
                                         top_k=40,
                                         temperature=0.7)[0].cpu()
        generated_string = self.tokenizer.decode(
            output_ids[input_len:], skip_special_tokens=True)

        reward = self.compute_sequence_score([generated_string])[0]
        self.archive.archive[self.archive.seed_instruction] = reward
        self.seed_instruction_reward = reward
        self.archive.seed_instruction_output = generated_string
        self.archive.output_list.append(generated_string)
        print("Orginal instruction reward = ", reward)

    def random_replacement(self,
                           states_score_list: List[float],
                           states: List[str],
                           state_complete: List[bool],
                           top_k: int = 3):

        if top_k >= len(states):
            print("Top k must be lesser than the number of states")
            return

        top_k_indices = sorted(range(len(states_score_list)),
                               key=lambda i: states_score_list[i], reverse=True)[:top_k]

        self.winning_beam_replacement_history.append(top_k_indices)

        top_k_states = [states[i] for i in top_k_indices]
        if not self.winning_beam:
            # Initialize counter to keep track of the frequency of winning beams
            self.winning_beam = Counter(top_k_indices)
        else:
            for index in top_k_indices:
                self.winning_beam[index] = self.winning_beam.get(index, 0)+1
        #print(self.winning_beam)

        output_states = []

        # Randomly replace the other states with one of the top k states
        for i in range(len(states)):
            if state_complete[i] or i in top_k_indices:
                output_states.append(states[i])
            else:
                output_states.append(random.choice(top_k_states))
        # Check if replacement cause some states to finish
        for i in range(len(state_complete)):
            if self.tokenizer.eos_token in output_states[i]:
                state_complete[i] = True
        return output_states, state_complete

    @torch.inference_mode
    def explore(self,
                augmented_instructions: List[str],
                previous_states: List[str],
                state_complete: List[bool],
                replacement_period: int = 40,
                show_generation_process: bool = False,
                look_ahead: int = 0
                ):

        candidate_states = []
        inp_len = []

        for i in range(len(state_complete)):
            # Have to check again because replacement may cause some state to be completed
            if self.tokenizer.eos_token in previous_states[i]:
                state_complete[i] = True
            if not state_complete[i]:
                # Continue exploring from previous state
                new_state = augmented_instructions[i]+previous_states[i]
                candidate_states.append(new_state)
                inp_len.append(
                    len(self.tokenizer.encode(augmented_instructions[i])))

        inputs = self.tokenizer(
            candidate_states, padding=True, return_tensors='pt')
        # Compute how many pad tokens each sequence is added
        pad_length = torch.sum(inputs['attention_mask'] == 0, axis=1)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(**inputs,
                                  max_new_tokens=replacement_period,
                                  do_sample=True,
                                  temperature=0.7,
                                  top_k=40)

        j = 0
        for i in range(len(previous_states)):
            if not state_complete[i]:
                output_ids = out[j, inp_len[j]+pad_length[j]:]
                generated_text = self.tokenizer.decode(
                    output_ids, skip_special_tokens=False)
                if len(output_ids) > 2048 or self.tokenizer.eos_token in generated_text:
                    state_complete[i] = True

                previous_states[i] = generated_text

                j += 1

        # Score the the generated text.
        if look_ahead > 0 and sum(state_complete) != len(state_complete):
            # If perform lookahead, continue exploring for lookahead tokens from the previous states
            print("Computing lookahead!")
            score_list = self.compute_sequence_score_lookahead(augmented_instructions,
                                                               previous_states,
                                                               state_complete,
                                                               look_ahead)
           
        else:

            score_list = self.compute_sequence_score(previous_states)

        if show_generation_process:
            for s in previous_states:
                print(s)
        return score_list, previous_states, state_complete

    @torch.inference_mode()
    def generate(self,
                 instruction: str,
                 iteration: int = 1,
                 replacement_period: int = 40,
                 show_generation_process: bool = True,
                 top_k: int = 3,
                 look_ahead: int = 0,
                 do_replacement: bool = True,
                 n_mutation: int = 1
                 ):
        
        self.winning_beam = []
        self.winning_beam_data = []
        self.compute_original_instruction_score()
        # Set best output to seed output
        previous_best_output = self.archive.seed_instruction_output
        previous_best_reward = self.seed_instruction_reward

        for i in range(iteration):

            print(f"Iteration {i+1}")
            all_instructions = self.archive.get_all_instructions()
            previous_instruction = random.sample(all_instructions, 1)[0]
            previous_instruction_reward = self.archive.archive[previous_instruction]

            mutated_instructions = self.mutate(
                previous_instruction, n=n_mutation)
            #print(mutated_instructions)
            # Somehow mutation failed, so we do not mutate and use the current instruction
            if len(mutated_instructions) != 5*n_mutation:
                print("Mutation failed!")
                mutated_instructions = [
                    previous_instruction for i in range(5*n_mutation)]

            if "mistral" in self.model_name:
                augmented_instructions = [apply_mistral_instruct_template(
                    aug, self.tokenizer) for aug in mutated_instructions]
            if 'Llama-3' in self.model_name:
                augmented_instructions = [apply_llama3_template(
                    aug, self.tokenizer) for aug in mutated_instructions]

            # Initialize all states to empty string
            previous_states = ['' for i in range(len(augmented_instructions))]
            state_complete = [False for i in range(
                len(augmented_instructions))]

            while True:
                if do_replacement:

                    score_list, previous_states, state_complete = self.explore(augmented_instructions,
                                                                               previous_states,
                                                                               state_complete,
                                                                               replacement_period=replacement_period,
                                                                               show_generation_process=show_generation_process,
                                                                               look_ahead=look_ahead
                                                                               )

                    previous_states, state_complete = self.random_replacement(states_score_list=score_list,
                                                                              states=previous_states,
                                                                              state_complete=state_complete,
                                                                              top_k=top_k)
                else:
                    # If no replacement, explore for max 2048 tokens
                    score_list, previous_states, state_complete = self.explore(augmented_instructions,
                                                                               previous_states,
                                                                               state_complete,
                                                                               replacement_period=2048,
                                                                               show_generation_process=show_generation_process,
                                                                               look_ahead=look_ahead
                                                                               )
                    # Force all states to be complete after 2048 tokens
                    state_complete = [True for _ in range(len(state_complete))]

                if sum(state_complete) == len(state_complete):
                    # Clean up all the pad and eos tokens
                    previous_states = [self.tokenizer.decode(self.tokenizer.encode(previous_states[i]),
                                                             skip_special_tokens=True) for i in range(len(previous_states))]

                    reward_list = self.compute_sequence_score(previous_states)
                    index = np.argmax(reward_list)
                    best_output = previous_states[index]

                    beams_index_sorted = np.argsort(
                        list(self.winning_beam.values()))
                    winning_beam = dict(self.winning_beam)
                    top_2_beams_index = sorted(
                        winning_beam, key=winning_beam.get, reverse=True)[:2]

                    top_2_augmented_instruction = [
                        mutated_instructions[i] for i in top_2_beams_index]
                    top_2_reward = [reward_list[i] for i in top_2_beams_index]

                    self.archive.update_archive(
                        previous_instruction, top_2_augmented_instruction, top_2_reward)

                    self.winning_beam_data.append({"winning_beam_total_frequency": dict(self.winning_beam),
                                                   "winning_beam_replacement_cycle": self.winning_beam_replacement_history})
                    self.winning_beam = None
                    self.winning_beam_replacement_history = []

                    #print(max(reward_list))
                    # Compare with original instruction reward and return the greater one
                    if max(reward_list) > previous_best_reward:
                        previous_best_output = best_output
                        previous_best_reward = max(reward_list)

                    self.archive.output_list.append(previous_best_output)

                    break
        return previous_best_output
