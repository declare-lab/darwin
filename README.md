# Inference Time Alignment with Reward-Guided Tree Search

DARWIN is a inference-time alignment technique that uses a reward-guided tree search framework to align the LLM and achieve comparable performance to preference optimization on AlpacaEval and MT Bench.

Paper Link: https://arxiv.org/abs/2406.15193

# How to use?

To run darwin, check out the demo notebook. You can run darwin with just a few lines of code!

To run evaluation on alpaca eval benchmark, you can use the following command

```
python3 alpaca_generate.py --method='darwin'  --model_name='meta-llama/Meta-Llama-3-8B-Instruct'  --replacement_period=40 --iteration=3 --n_mutation=1
```

The results will be saved in a json file where the 'past_outputs' contains a list of outputs for original output and mutation cycle 1, 2, 3. Please format the output into the alpaca_eval format from https://github.com/tatsu-lab/alpaca_eval

# Citation

If you use Darwin in your publication, please cite it by using the following BibTeX entry.

```@misc{hung2024inferencetimealignmentrewardguided,
      title={Inference Time Alignment with Reward-Guided Tree Search},
      author={Chia-Yu Hung and Navonil Majumder and Ambuj Mehrish and Soujanya Poria},
      year={2024},
      eprint={2406.15193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15193},
}
```
