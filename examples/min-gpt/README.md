# Fine Tuning GPT Models using MinGPT

This repo is an adaptation of Andrej Karpathy's [MinGPT project](https://github.com/karpathy/minGPT). It uses the `@torchrun` decorator with `@kubernetes` on Metaflow to train a MinGPT model with distributed training. 

Many of the files in this example have been directly sourced from the MinGPT project with minimal or no adjustments. The [gpt2_train_cfg.yaml](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/gpt2_train_cfg.yaml), [char_dataset.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/char_dataset.py), [model.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py), [trainer.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py), [main.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/main.py) have been sourced from the MinGPT project. The `flow.py` and `flow_oss.py` uses the minGPT's CLI script via Metaflow's `@torchrun` decorator.

## Running with Open source Metaflow on Kubernetes
- `python flow_oss.py run` 

## Running on the Outerbounds Platform
- `python flow.py --environment=fast-bakery run`

