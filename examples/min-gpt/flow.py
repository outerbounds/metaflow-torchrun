from gpu_profile import gpu_profile
from metaflow import FlowSpec, step, torchrun, current, kubernetes, pypi


class MinGPT(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=2)

    @gpu_profile(interval=1)
    @kubernetes(
        image="registry.hub.docker.com/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        cpu=12,
        gpu=1,
        memory=28000,
        shared_memory=8000,
    )
    @pypi(
        packages={
            "fsspec": "2024.12.0",
            "hydra-core": "1.3.2",
            "omegaconf": "2.3.0",
            "aiohttp": "3.11.11",
            "requests": "2.32.3",
            "matplotlib": "3.10.0"
        }
    )
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(entrypoint="main.py", nproc_per_node=1)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MinGPT()
