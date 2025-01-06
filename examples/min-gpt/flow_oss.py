from gpu_profile import gpu_profile
from metaflow import FlowSpec, IncludeFile, step, torchrun, current, kubernetes


num_gpus: int = 2


class MinGPT(FlowSpec):
    config_file = IncludeFile(
        name="config_file",
        is_text=True,
        help="gpt2 config file.",
        default="./gpt2_train_cfg.yaml",
    )

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=2)

    @gpu_profile(interval=1)
    @kubernetes(
        image="registry.hub.docker.com/eddieob/min-gpt:3",
        cpu=12,
        gpu=num_gpus,
        memory=28000,
        shared_memory=8000,
    )
    @torchrun
    @step
    def torch_multinode(self):
        with open("config.yaml", "w") as fp:
            fp.write(self.config_file)
        current.torch.run(entrypoint="main.py", nproc_per_node=num_gpus)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MinGPT()
