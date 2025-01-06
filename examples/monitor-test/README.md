# Introduction

The following two files showcase how to get started with `@torchrun` with `@kubernetes` on Metaflow.

1. `script.py` contains a basic code snippet for each process to print their rank and world size. It deliberately raises a `ValueError` so as to check if the nodes themselves fail when their health monitor realises a failure in a different node.

2. `flow.py` uses the above script via `current.torch.run(entrypoint="script.py")`.

- The flow can be run using `python flow.py run`.
