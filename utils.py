from transformers import AutoTokenizer
from typing import List


class Archive:

    def __init__(self, seed_instruction):
        self.archive_size = 1
        self.seed_instruction = seed_instruction
        self.archive = {}
        self.past_mutation = {}
        self.previous_best_instruction = seed_instruction
        self.output_list = []
        self.seed_instruction_output = None

    def set_seed_instruction(self, instruction: str):
        self.seed_instruction = seed_instruction

    def get_all_instructions(self):
        return list(self.archive.keys())

    def update_archive(self, previous_instruction: str,
                       mutated_instruction_list: List[str],
                       mutated_instruction_reward_list: List[float]):
        # Each instruction should be mutated into multiple instruction and return top k mutated instruction
        # Update archive will only be once called every iteration
        previous_instruction_reward = self.archive[previous_instruction]
        updated = False
        for i in range(len(mutated_instruction_list)):
            mutated_instruction = mutated_instruction_list[i]
            mutated_instruction_reward = mutated_instruction_reward_list[i]
            if previous_instruction_reward < mutated_instruction_reward:
                # Remove old instruction if mutated instruction has a higher reward
                print(
                    f"Previous instruction replaced with {mutated_instruction}", f"New reward: {mutated_instruction_reward}")
                self.archive[mutated_instruction] = mutated_instruction_reward
                if mutated_instruction != previous_instruction:
                    updated = True

                if (previous_instruction, previous_instruction_reward) in self.past_mutation:
                    # Keep track of past mutation and reward
                    self.past_mutation[(previous_instruction, previous_instruction_reward)].append(
                        (mutated_instruction, mutated_instruction_reward))
                else:
                    self.past_mutation[(previous_instruction, previous_instruction_reward)] = [
                        (mutated_instruction, mutated_instruction_reward)]
        if updated:
            self.archive.pop(previous_instruction, None)


def apply_mistral_instruct_template(query: str, tokenizer):

    messages = [
        {"role": "user", "content": query}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    return tokenizer.decode(inputs, skip_special_tokens=False)


def apply_llama3_template(query: str, tokenizer):

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": query}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    return tokenizer.decode(inputs, skip_special_tokens=False)
