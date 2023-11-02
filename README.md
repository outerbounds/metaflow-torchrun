# Metaflow torchrun decorator

### Introduction
This repository implements a plugin to run parallel Metaflow tasks as nodes in a [torchrun](https://pytorch.org/docs/stable/elastic/run.html) job which can be submitted to AWS Batch or a Kubernetes cluster.

### Features
- <b>Automatic torchrun integration:</b> This extension provides a simple and intuitive way to incorporate PyTorch distributed programs in your Metaflow workflows using the `@torchrun` decorator
- <b>No changes to model code:</b> The `@torchrun` decorator exposes a new method on the Metaflow current object, so you can run your existing torch distributed programs inside Metaflow tasks with no changes in the research code.
- <b>Run one command:</b> You don't need to log into many nodes and run commands on each. Instead, the `@torchrun` decorator will select arguments for the torchrun command based on the requests in Metaflow compute decorators like number of GPUs. Network addresses are automatically discoverable. 
- <b>No user-facing subprocess calls:</b> At the end of the day, `@torchrun` is calling a subprocess inside a Metaflow task. Although many Metaflow users do this, it can make code difficult to read for beginners. One major goal of this plugin is to motivate hardening and automating a pattern for submitting subprocess calls inside Metaflow tasks.

### Installation
You can install it with:
```
pip install metaflow-torchrun
```

### Getting Started
And then you can import it and use in parallel steps:
```
from metaflow import FlowSpec, step, torchrun

...
class MyGPT(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(cpu=N_CPU, gpu=N_GPU, memory=MEMORY)
    @torchrun
    @step
    def torch_multinode(self):
        ...
        current.torch.run(
            entrypoint="main.py", # No changes made to original script.
            entrypoint_args = {"main-arg-1": "123", "main-arg-2": "777"},
            nproc_per_node=1,     # edge case of a torchrun arg user-facing.
        )
        ...
    ...
```

### Examples

| Directory | torch script description |
| :--- | ---: |
| [Hello](examples/hello/README.md) | Each process prints their rank and the world size. |  
| [Tensor pass](examples/tensor-pass/README.md) | Main process passes a tensor to the workers. |  
| [Torch DDP](examples/torch-ddp/README.md) | A flow that uses a [script from the torchrun tutorials](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html) on multi-node DDP. |  
| [MinGPT](examples/min-gpt/README.md) | A flow that runs a [torchrun GPT demo](https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html) that simplifies [Karpathy's minGPT](https://github.com/karpathy/minGPT) in a set of parallel Metaflow tasks each contributing their `@resources`. |

### License 
`metaflow-torchrun` is distributed under the <u>Apache License</u>.
