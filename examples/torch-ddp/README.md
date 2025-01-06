# Introduction

The following three files showcase how to run distributed model training with the DDP i.e. distributed data parallelism approach with `@torchrun` and `@kubernetes` on Metaflow.

1. `datautils.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/datautils.py)

2. `multinode_trainer.py` is adapted from [here](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py)

3. `flow.py` uses the above script via `current.torch.run(entrypoint="multinode_trainer.py")` along with some entrypoint arguments such as total epochs, batch size, etc.

- The flow can be run using `python flow.py run`.
