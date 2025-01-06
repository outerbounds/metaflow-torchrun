# Introduction

The following two files showcase how to get started with `@torchrun` with `@kubernetes` on Metaflow.

1. `hi-torchrun.py` contains a basic code snippet for each process to print their rank and world size.

2. `flow.py` uses the above script via `current.torch.run(entrypoint="hi-torchrun.py")`.

- The flow can be run using `python flow.py run`.
