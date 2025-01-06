# Introduction

The following two files showcase how a main process can pass a tensor to workers using `@torchrun` with `@kubernetes` on Metaflow.

1. `script.py` contains the code snippet for passing the tensor from main to worker process.

2. `flow.py` uses the above script via `current.torch.run(entrypoint="script.py")`.

- The flow can be run using `python flow.py run`.
