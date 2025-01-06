import os
import socket
import argparse
import torch
import torch.distributed as dist


def check_env_var_exists(var_name, dtype=str):
    return None if not var_name in os.environ else dtype(os.environ[var_name])


# set by torchrun
LOCAL_RANK: int = check_env_var_exists("LOCAL_RANK", int)
WORLD_SIZE: int = check_env_var_exists("WORLD_SIZE", int)

# set by jobset
WORLD_RANK: int = check_env_var_exists("RANK", int)
MASTER_ADDR: str = check_env_var_exists("MASTER_ADDR", str)


def get_device(backend="gloo"):
    device = "cpu"
    if backend == "nccl" and torch.cuda.is_available():
        device = "cuda:{}".format(LOCAL_RANK)
    elif torch.cuda.is_available():
        print(
            "Found GPU, but NCCL backend not selected, using gloo. NCCL is recommended for GPU IPC."
        )
        device = "cuda:{}".format(LOCAL_RANK)
    if torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)


def send_tensor(backend):
    # Wait until all processes reach this point.
    device = get_device(backend)
    if device.type == "cuda":
        # Specify device_ids for barrier when using CUDA
        torch.distributed.barrier(device_ids=[device.index])
    else:
        torch.distributed.barrier()

    # Make a tensor.
    # Notice this happens in all processes.
    tensor = torch.zeros(2, 3).to(device)

    # On the main process,
    if WORLD_RANK == 0:
        # add a new tensor to the previous one.
        tensor += torch.rand(2, 3).to(device)

        # Send the new tensor to all other processes.
        ## dist.send() is blocking by default - it will wait for each send to complete
        ## before moving to the next one
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
        print("control sent {}".format(tensor))

    else:
        # Receive the tensor from process 0.
        dist.recv(tensor=tensor, src=0)
        print("worker_{} has received {} from rank {}\n".format(WORLD_RANK, tensor, 0))


def run(backend):

    print(
        "Hello from process {} (out of {}) on {} with rank {} and local rank {}".format(
            WORLD_RANK, WORLD_SIZE, socket.gethostname(), WORLD_RANK, LOCAL_RANK
        )
    )

    # A magic torch function to ensure processes can coordinate with master.
    dist.init_process_group(
        backend,
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        init_method=f"env://",
    )

    # Call sample end user torch distributed code.
    send_tensor(backend)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default='nccl')
    args = parser.parse_args()
    run(backend=args.backend)
