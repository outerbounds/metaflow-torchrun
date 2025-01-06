# Fine Tuning GPT Models using MinGPT

This repo is an adaptation of Andrej Karpathy's [MinGPT project](https://github.com/karpathy/minGPT). It uses the `@torchrun` decorator with `@kubernetes` on Metaflow to train a MinGPT model with distributed training. 

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow.py`

2. `gpt2_train_cfg.yaml` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/gpt2_train_cfg.yaml)

3. `char_dataset.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/char_dataset.py)

4. `model.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py)

5. `trainer.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py)

6. `main.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/main.py)

7. `flow.py` and `flow_oss.py` uses the above script via `current.torch.run(entrypoint="main.py")`.

8. `Dockerfile` was used to build the `eddieob/min-gpt:3` image, used in the `flow_oss.py` file.

- The flow can be run using `python flow_oss.py run` if using the OSS version with the `eddieob/min-gpt:3` docker image.
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python flow.py --environment=fast-bakery run`
