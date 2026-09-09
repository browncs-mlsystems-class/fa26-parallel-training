import os, sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.multiprocessing as mp

import utils
from ddp import train_vgg16_cifar10_ddp
from fsdp import train_vgg16_cifar10_fsdp
from model_parallel import train_vgg16_cifar10_model_parallel
from pipeline_parallel import train_vgg16_cifar10_pipeline_parallel
from correctness_test import check_weights_test, check_outputs_test

def train_vgg16_cifar10(
    num_batches: int = 10, batch_size: int = 64, 
    learning_rate: float = 0.001, remove_avg_pool: bool = False, verbose: bool = False
) -> tuple[nn.Module, torch.Tensor]:
    """
    Trains and evaluates a VGG16 model on the CIFAR-10 dataset without any parallelism techniques.
    
    Args:
        num_batches (int): the number of batches to train for; defaults to 10
        batch_size (int): the number of data points processed in one epoch of
        training; defaults to 64
        learning_rate (float): the learning rate of the optimizer; defaults to 0.001
        remove_avg_pool (bool): whether to train with the average pooling layer removed
        from VGG16
        verbose (bool): whether to provide information print statements about training;
        defaults to False
    Returns:
        tuple[nn.Module, torch.Tensor]: the trained model and the output tensor of the
        model of the next batch in the training data loader
    """
    device = torch.device("cpu")

    # Load CIFAR-10 dataset
    train_dataset = utils.get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Load the VGG16 model
    utils.seed_everything(1390)
    model = vgg16(weights=None)

    # Modify the classifier to fit CIFAR-10 (10 classes)
    if remove_avg_pool:
        model.classifier[0] = nn.Linear(512, 4096)
    model.classifier[6] = nn.Linear(4096, 10)
    if remove_avg_pool:
        layers = [layer for layer in model.features] + [nn.Flatten()] + [layer for layer in model.classifier]
    else:
        layers = [layer for layer in model.features] + [model.avgpool, nn.Flatten()] + [layer for layer in model.classifier]
    model = nn.Sequential(*layers)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if verbose:
        print(f"Starting training for {num_batches} batches...")

    # Training loop
    model.eval()
    for i, (inputs, labels) in enumerate(train_loader):
        if num_batches == i:
            output_tensor = model(inputs)
            break
        start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # zero the parameter gradients

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        end = time.time()
        if verbose:
            print(f"Batch {i + 1}/{num_batches}:\n"
                f" - batch loss: {loss.item():.3f}\n"
                f" - time: {end - start:.3f} sec")

    if verbose:
        print("Finished training")
    return model, output_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS1390 Project 1: Parallelism Techniques")
    parser.add_argument(
        "parallelism", type=str,
        choices=["none", "ddp", "fsdp", "model", "pipeline"],
        default="none",
        help=(
            "Type of parallelism to use. Options are: "
            "'none' (no parallelism), "
            "'ddp' (Distributed Data Parallel), "
            "'fsdp' (Fully Sharded Data Parallel), "
            "'model' (Model Parallelism), "
            "'pipeline' (Pipeline Parallelism)."
        ),
    )
    parser.add_argument("--cores_per_worker", "-c", type=int, default=1,
                        help="Number of CPU cores to use per worker.")
    parser.add_argument("--num_workers", "-w", type=int, default=1, 
                        help="Number of distributed workers.")
    parser.add_argument("--num_batches", "-n", type=int, default=5, 
                        help="Number of batches to train for")
    parser.add_argument("--num_microbatches", "-m", type=int, default=2,
                        help="Number of microbatches to split each batch into (only for pipeline parallel)")
    parser.add_argument("--batch_size", "-b", type=int, default=64, 
                        help="The size of each batch to use when training")
    parser.add_argument("--learning_rate", "-l", type=float, default=0.001, 
                        help="The learning rate of the optimizer when training")
    parser.add_argument("--check_weights", action="store_true", 
                        help="Check whether the parallelized training results in identical weight updates to a non-parallelized PyTorch implementation.")
    parser.add_argument("--check_output", action="store_true",
                        help="Check whether the output of a single batch matches exactly with a non-parallelized PyTorch implementation.")
    mp.set_start_method("spawn")
    
    args = parser.parse_args()

    for path in ('./state_dicts', './saved_tensors'):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            utils.clear_dir(path)
    
    stats = None
    utils.seed_everything(1390)
    if args.parallelism == "ddp":
        print(f"Training with Distributed Data Parallel (DDP) "
              f"with {args.num_workers} worker{'s' if args.num_workers != 1 else ''}, "
              f"with {args.cores_per_worker} core{'s' if args.cores_per_worker != 1 else ''} per worker, "
              f"and {args.num_batches} batch{'es' if args.num_batches != 1 else ''}")
        stats = train_vgg16_cifar10_ddp(
            world_size=args.num_workers,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cores_per_rank=args.cores_per_worker,
            check_weights=args.check_weights,
            check_output=args.check_output
        )
    elif args.parallelism == "fsdp":
        print("Training with Fully Sharded Data Parallel (FSDP) "
              f"with {args.num_workers} worker{'s' if args.num_workers != 1 else ''}, "
              f"with {args.cores_per_worker} core{'s' if args.cores_per_worker != 1 else ''} per worker, "
              f"and {args.num_batches} batch{'es' if args.num_batches != 1 else ''}")
        stats = train_vgg16_cifar10_fsdp(
            world_size=args.num_workers,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cores_per_rank=args.cores_per_worker,
            check_weights=args.check_weights,
            check_output=args.check_output
        )
    elif args.parallelism == "model":
        print("Training with Model Parallelism "
              f"with {args.num_workers} worker{'s' if args.num_workers != 1 else ''}, "
              f"with {args.cores_per_worker} core{'s' if args.cores_per_worker != 1 else ''} per worker, "
              f"and {args.num_batches} batch{'es' if args.num_batches != 1 else ''}")
        stats = train_vgg16_cifar10_model_parallel(
            world_size=args.num_workers,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cores_per_rank=args.cores_per_worker,
            check_output=args.check_output
        )
    elif args.parallelism == "pipeline":
        print("Training with Pipeline Parallelism "
              f"with {args.num_workers} worker{'s' if args.num_workers != 1 else ''}, "
              f"with {args.cores_per_worker} core{'s' if args.cores_per_worker != 1 else ''} per worker, "
              f"and {args.num_batches} batch{'es' if args.num_batches != 1 else ''}")
        stats = train_vgg16_cifar10_pipeline_parallel(
            world_size=args.num_workers,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cores_per_rank=args.cores_per_worker,
            num_microbatches=args.num_microbatches,
            check_output=args.check_output
        )
    else:
        print("Training without parallelism")
        train_vgg16_cifar10(
            num_batches=args.num_batches, batch_size=args.batch_size, 
            learning_rate=args.learning_rate, verbose=True
        )
        sys.exit(0)

    # Perform correctness tests, if specified
    if args.check_weights and args.parallelism in ('ddp', 'fsdp'):
        print("\nTraining baseline model for weights correctness comparison...")
        model, _ = train_vgg16_cifar10(
            num_batches=args.num_batches, batch_size=args.batch_size*args.num_workers, 
            learning_rate=args.learning_rate
        )
        check_weights_test(model.state_dict(), './state_dicts')  
    if args.check_output:
        print("\nTraining baseline model for output correctness comparison...")
        if args.parallelism in ('ddp', 'fsdp'):
            _, baseline_output = train_vgg16_cifar10(
                num_batches=args.num_batches, batch_size=args.batch_size*args.num_workers, 
                learning_rate=args.learning_rate, remove_avg_pool=False
            )
        else:
            _, baseline_output = train_vgg16_cifar10(
                num_batches=args.num_batches, batch_size=args.batch_size, 
                learning_rate=args.learning_rate, remove_avg_pool=True
            )
        check_outputs_test(baseline_output, './saved_tensors', args.parallelism)

    # Process and save generated statistics
    stats[utils.NUM_WORKERS] = args.num_workers
    stats[utils.NUM_BATCHES] = args.num_batches
    stats[utils.BATCH_SIZE] = args.batch_size
    stats[utils.LEARNING_RATE] = args.learning_rate
    stats[utils.CORES_PER_WORKER] = args.cores_per_worker
    os.makedirs(utils.DATA_DIR, exist_ok=True)
    output_path = f"{utils.DATA_DIR}/stats_{args.parallelism}{args.num_workers}.json"
    with open(f"{utils.DATA_DIR}/stats_{args.parallelism}{args.num_workers}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics of this run to {output_path}")
