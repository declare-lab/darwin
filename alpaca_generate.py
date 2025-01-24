import argparse
from tqdm import tqdm
import datasets
import json
from darwin import Darwin, BestOfN
from utils import Archive
import time


def parse_range(range_str):
    try:
        start, end = map(int, range_str.split('-'))
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Range must be in the format start-end (e.g., 500-600)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some dataset and save results.")

    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the results')
    parser.add_argument('--model_name', type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2', help='Model to peform strategy on')
    parser.add_argument('--method', type=str,
                        default='darwin', help='Strategy')

    parser.add_argument('--iteration',
                        type=int, default=1, help="Number of iterations for darwin"
                        )
    parser.add_argument('--replacement_period', type=int, default=40, help="Range to process in the format start-end (e.g., 500-600)"
                        )
    parser.add_argument("--reward_model", type=str,
                        default='sfairXC/FsfairX-LLaMA3-RM-v0.1',
                        help="Reward model to perform reward calculation. If you are changing the reward model, please check the corresponding chat template and modify the code accordingly")

    parser.add_argument("--n", type=int, default=5,
                        help="Number of beams when performing best of N")
    parser.add_argument("--look_ahead", type=int, default=0,
                        help="Number of lookahead tokens when performing reward calculation")
    parser.add_argument("--n_mutation", type=int, default=1,
                        help="Number of mutation to perform. Each mutation generates 5 mutation. If n_mutation=3, this will generate 15 mutation, leading to 15 beams in the search process")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = "cuda"

    model_name = args.model_name
    method = args.method
    look_ahead = args.look_ahead
    n = args.n
    n_mutation = args.n_mutation
    reward_model_name = args.reward_model

    if method == 'best_of_n':
        generator = BestOfN(
            model_name, reward_model_name=reward_model_name, device=device)
    elif method == 'darwin' or 'mutation_no_replacement':
        generator = Darwin(
            model_name, reward_model_name=reward_model_name, device=device)

    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    outputs = []
    iteration = args.iteration
    replacement_period = args.replacement_period
    exp_id = str(int(time.time()))
    saved_model = model_name.split('/')[-1]
    rm_save_model = reward_model_name.split('/')[-1]
    
    if method == 'best_of_n':
        save_name = f'{args.save_dir}/{exp_id}_alpaca_eval_{saved_model}_{method}_{n}_rm_{rm_save_model}_{start}-{end}.json'
    elif method == 'mutation_no_replacement':
        save_name = f'{args.save_dir}/{exp_id}_alpaca_eval_{saved_model}_{method}_iter{iteration}_rm_{rm_save_model}_{start}-{end}.json'
    elif method == 'darwin':
        save_name = f'{args.save_dir}/{exp_id}_alpaca_eval_{saved_model}_{method}_iter{iteration}_look_ahead{look_ahead}_rm_{rm_save_model}_{replacement_period}_{start}-{end}.json'

    print(f"Saving to {save_name}!")
    for example in tqdm(eval_set):

        archive = Archive(example['instruction'])

        if method == 'darwin':
            generator.archive = archive

            res = generator.generate(example["instruction"],
                                     replacement_period=replacement_period,
                                     show_generation_process=True,
                                     iteration=iteration,
                                     look_ahead=look_ahead,
                                     n_mutation=n_mutation)
            past_mutations = {
                str(k): v for k, v in archive.past_mutation.items()}
            response = [{'instruction': example["instruction"],
                         'output': res,
                         'original_output': example['output'],
                         'generator': model_name,
                         'output_without_aug': archive.output_list[0],
                         'past_outputs': archive.output_list,
                         'past_mutation': past_mutations,
                         'winning_beams': generator.winning_beam_data}]

        elif method == 'best_of_n':

            res = generator.generate(example["instruction"], n=n)
            response = [{'instruction': example["instruction"],
                         'output': res,
                         'generator': model_name}]

        elif method == 'mutation_no_replacement':
            generator.archive = archive

            res = generator.generate(example["instruction"],
                                     replacement_period=replacement_period,
                                     show_generation_process=True,
                                     iteration=iteration,
                                     look_ahead=look_ahead,
                                     do_replacement=False)
            past_mutations = {
                str(k): v for k, v in archive.past_mutation.items()}
            response = [{'instruction': example["instruction"],
                         'output': res, 'original_output': example['output'],
                         'generator': model_name,
                         'output_without_aug': archive.output_list[0],
                         'past_outputs': archive.output_list,
                         'past_mutation': past_mutations}]

        outputs += response

        with open(f'{save_name}', 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
