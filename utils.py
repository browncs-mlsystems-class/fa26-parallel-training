import os
import random
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import time 

DEBUG_PRINT = True # whether to print verbose debugging output

# Identifiers for dictionary key values
# For communication information
ACTIVATION_SIZE = 'activation_size' 
GRADIENT_SIZE = 'gradient_size'

# For per-rank statistics
RANK = 'rank'
COMP_TIME = 'computation_time'
COMM_TIME = 'communication_time'
OPT_TIME = 'optimizer_update_time'
TOTAL_TIME = 'total_time'
BATCHES_TIMES = 'batches_times'

# For overall statistics
CORES_PER_WORKER = 'cores_per_worker'
NUM_WORKERS = 'num_workers'
NUM_BATCHES = 'num_batches'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'

DATA_DIR = "./data"


def log_memory(device: torch.device,
               rank: int,
               batch_idx: int,
               layer_idx: int,
               phase: str):
    """
    Returns an entry representing the current memory usage
    on this rank.
    Can be used to build a timeseries of memory usage.
    If not using the gpu, returns an empty entry.
    """
    if not(torch.cuda.is_available()):
        return {}
    sync_if_cuda(device)
    entry = {"batch": batch_idx, 
             "layer": layer_idx,
             "rank": rank,
             "phase": phase,
             "allocated_bytes": torch.cuda.memory_allocated(device),
             "peak_bytes": torch.cuda.max_memory_allocated(device),
             "timestamp": time.time()}
    mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Rank {rank}, b {batch_idx}, layer {layer_idx}, phase {phase}, mem: {mem} MB")
    return entry

def sync_if_cuda(device: torch.device | None = None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

def debug_print(*args, **kwargs):
    """
    Togglable printing, according to the `DEBUG_PRINT` flag in `utils.py`.
    """
    if DEBUG_PRINT:
        print(*args, **kwargs)

def clear_dir(path: str):
    """
    Clears the given directory of all saved PyTorch files.
    
    Args:
        path (str): the path of the given directory to clear
    """
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if filename.endswith(".pt"):
            os.remove(filepath)

def get_train_dataset() -> Dataset:
    """
    Loads and preprocesses the CIFAR-10 training and test dataset.

    Returns:
        Dataset: the training dataset and the test dataset
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    return torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)


def seed_everything(s: int):
    """
    This function allows us to set the seed for all of our random functions
    so that we can get reproducible results.

    Args:
        s (int): the seed to seed all random functions with
    """
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    # one of the ops (avg pool2d) does not have a determinstic impl
    # in CuBLAS
    torch.use_deterministic_algorithms(True, warn_only=True)

def agg_stats_per_rank(stats_queue: mp.Queue) -> dict[int, dict]:
    """
    Aggregates the stats from all ranks, passed via the given queue, into
    a single dictionary.
    
    Args:
        stats_queue (mp.Queue): the queue used to communicate each rank's 
        statistics to the main process
    Returns:
        dict[int, dict]: a dictionary from each rank to the rank's statistics
    """
    stats = {}
    while True:
        try:
            rank_stats = stats_queue.get(block=False)
        except queue.Empty:
            break
        stats[rank_stats['rank']] = rank_stats
    return stats


# ---------------------------------------------------------------------------------
# Parallelism training helpers

def parallel_setup(rank: int, world_size: int):
    """
    Performs any generic set-up for parallel training for the current process.
    
    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    use_cuda = torch.cuda.is_available()
    debug_print(f"Running setup on rank {rank}, world_size {world_size}, use_cuda {use_cuda}")
    backend="gloo"
    if use_cuda:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        backend="nccl"

    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(rank)
    # initialize the process group
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.autograd.set_detect_anomaly(True)
        return device
    except Exception as e:
        raise RuntimeError(f"Could not init process group: {e}") from e

def pin_to_core(rank: int, num_cores: int):
    """
    Pins the current process to a specific number of unique cores.
    
    Args:
        rank (int): the rank of the current process
        num_cores (int): the number of cores to pin the current process to
    """
    start_core = rank * num_cores
    end_core = start_core + num_cores
    cores_print = f"core{f' {start_core}' if num_cores == 1 else f's {start_core}-{end_core-1}'}"
    debug_print(f"Rank {rank} pinning to {cores_print}")
    os.sched_setaffinity(0, range(start_core, end_core))
    debug_print(f"Rank {rank} pinned to {cores_print}")

def parallel_cleanup():
    """
    Performs any generic cleaning up for parallel training for the current process.
    ALL work for the process should be performed before this method is called.
    """
    dist.destroy_process_group()


# For model and pipeline parallelism
def analyze_communication_with_partitions(
    input_size: torch.Size, partitions: list[list[nn.Module]]
) -> list[dict[str, torch.Size]]:
    """
    Analyze and return communication requirements for the partition of a model, given said
    partitions.
    
    Args:
        input_tensor (torch.Size): the size of an input tensor to the model
        partitions (list[list[nn.Module]]): the partitioning of the layers of the
        model, where each element of the list is a partition, i.e. a list of layers        
    Returns:
        list[dict[str, int]]: the results of communication analysis; for 
        each corresponding partition in the inputted list, a mapping from:
         - `ACTIVATION_SIZE` to the size of the activation that that partition
           receives from the previous partition
         - `GRADIENT_SIZE` to the size of the gradient that it receives from
           the next partition
    """
    debug_print(f"Analyzing communication information between {len(partitions)} partitions of VGG16 model")
    communication_info: list[dict[str, torch.Size]] = []

    # TODO: Implement!
    
    return communication_info

# For model and pipeline parallelism
def split_vgg16(model: nn.Module, world_size: int) -> list[list[nn.Module]]:
    """
    Splits a VGG16 model into partitions for model parallelism for the specified 
    world size.
    
    Args:
        model (nn.Module): the VGG16 model instance
        world_size (int): the number of partitions to split the model into
        
    Returns:
        list[list[torch.nn.Module]]: a list of partitions, where each partition 
        is a list of layers
    """
    debug_print(f"Splitting VGG16 into {world_size} partition{'s' if world_size > 1 else ''}...")
    if world_size < 1 or world_size > 6:
        raise ValueError("Supported world_size values are 1, 2, 3, 4, 5, and 6.")
    
    # Extract the convolutional blocks and fully connected layers
    conv_blocks = [
        model.features[:4],     # Conv1: Conv(64) → Conv(64) → Pool
        model.features[4:9],    # Conv2: Conv(128) → Conv(128) → Pool
        model.features[9:16],   # Conv3: Conv(256) → Conv(256) → Conv(256) → Pool
        model.features[16:23],  # Conv4: Conv(512) → Conv(512) → Conv(512) → Pool
        model.features[23:],    # Conv5: Conv(512) → Conv(512) → Conv(512) → Pool
    ]
    fc_layers = [
        nn.Flatten(),  # Flatten
    	model.classifier[0],  # FC(4096)
    	model.classifier[1],  # ReLU
    	model.classifier[2],  # Dropout (optional)
    	model.classifier[3],  # FC(4096)
    	model.classifier[4],  # ReLU
    	model.classifier[5],  # Dropout (optional)
    	model.classifier[6],  # FC(1000 or 10 for CIFAR10)
    ]
    
    # Combine convolutional and fully connected layers
    all_blocks = conv_blocks + fc_layers
    
    # Partition sizes based on world_size
    partition_sizes = {
        1: [len(all_blocks)],  # Single partition contains all layers
        2: [5, len(all_blocks) - 5],  # 5 blocks for conv, rest for fc
        3: [2, 3, len(all_blocks) - 5],  # Split conv blocks and fc layers
        4: [2, 2, 3, len(all_blocks) - 7],  # Further split conv blocks
        5: [1, 2, 2, 3, len(all_blocks) - 8],  # Split first conv blocks finely
        6: [1, 1, 2, 2, 2, len(all_blocks) - 8],  # Maximum granularity
    }
    
    # Create partitions based on sizes
    sizes = partition_sizes[world_size]
    partitions = []
    start = 0
    for size in sizes:
        partitions.append(all_blocks[start:start + size])
        start += size
    
    for partition_id, partition in enumerate(partitions):
        debug_print(f"Partition {partition_id} - Layers: {len(partition)}")
    
    return partitions
