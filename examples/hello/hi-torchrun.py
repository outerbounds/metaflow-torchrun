import torch.distributed as dist
import torch

if __name__ == "__main__":

    # initialize the process group
    dist.init_process_group(backend="gloo")

    torch.distributed.barrier()

    # get the rank and size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # a simple way to print a message from each process
    print("hello from process %s of %s" % (rank + 1, world_size))

    # tear down the process group
    dist.destroy_process_group()