# Introduction

The following seven files showcase how to train minGPT inspired from [Karpathy](https://github.com/karpathy/minGPT) using the `@torchrun` decorator with `@kubernetes` on Metaflow.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow.py`

2. `gpt2_train_cfg.yaml` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/gpt2_train_cfg.yaml)

3. `char_dataset.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/char_dataset.py)


4. `model.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py)

5. `trainer.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py)

6. `main.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/main.py)

7. `flow.py` uses the above script via `current.torch.run(entrypoint="main.py")`.

- The flow can be run using `python flow.py --package-suffixes=.yaml --environment=fast-bakery run`
