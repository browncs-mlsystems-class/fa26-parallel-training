print("Loading PyTorch and model...")
import torch
from torchvision.models import vgg16

import utils

RED = "\x1B[0;31m"
GREEN = "\x1B[0;32m"
BLUE = "\x1B[0;34m"
CLEAR = "\x1B[0m"
FAILED_MSG = f"{RED}FAILED{CLEAR}"
PASSED_MSG = f"{GREEN}PASSED{CLEAR}"

TEST_SIZES: list[tuple[int, int, int, int]] = [(1, 3, 32, 32), (3, 3, 32, 32), (32, 3, 32, 32), (41, 3, 32, 32), (64, 3, 32, 32)]
COMM_SIZES: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = [
    [((3, 32, 32), (10,))],
    [((3, 32, 32), (512, 1, 1)), ((512, 1, 1), (10,))],
    [((3, 32, 32), (128, 16, 16)), ((128, 16, 16), (512, 1, 1)), ((512, 1, 1), (10,))],
    [((3, 32, 32), (128, 16, 16)), ((128, 16, 16), (512, 4, 4)), ((512, 4, 4), (4096,)), ((4096,), (10,))],
    [((3, 32, 32), (64, 32, 32)), ((64, 32, 32), (256, 8, 8)), ((256, 8, 8), (512, 1, 1)), ((512, 1, 1), (4096,)), ((4096,), (10,))],
    [((3, 32, 32), (64, 32, 32)), ((64, 32, 32), (128, 16, 16)), ((128, 16, 16), (512, 4, 4)), ((512, 4, 4), (512,)), ((512,), (4096,)), ((4096,), (10,))]
]

if __name__ == "__main__":
    utils.DEBUG_PRINT = False
    model = vgg16(weights=None)
    model.classifier[0] = torch.nn.Linear(512, 4096)
    model.classifier[6] = torch.nn.Linear(4096, 10)
    
    for world_size in range(1, 7):
        partitions = utils.split_vgg16(model, world_size)
        print(f"\nTesting with model split into {BLUE}{world_size} partition{'s' if world_size != 1 else ''}{CLEAR}:")
        for i, size in enumerate(TEST_SIZES):
            print(f"{i+1}. Input size of {size}: ", end='')
            comm_info = utils.analyze_communication_with_partitions(torch.Size(size), partitions)
            
            passed = True
            if len(comm_info) != len(COMM_SIZES[world_size-1]):
                print(f"{FAILED_MSG}\nExpected to receive a list of {len(COMM_SIZES[world_size-1])} dictionaries, received {len(comm_info)}")
                continue
            
            for part_id, (part_info, expected_sizes) in enumerate(zip(comm_info, COMM_SIZES[world_size-1])):
                if not isinstance(part_info, dict):
                    print(f"{FAILED_MSG}\nElement with index {part_id} of returned list was not a dict!")
                    passed = False
                    break
                if utils.ACTIVATION_SIZE not in part_info:
                    print(f"{FAILED_MSG}\nDictionary for partition {part_id+1}/{world_size} "
                          f"did not contain an entry with the key \"{utils.ACTIVATION_SIZE}\", "
                          f"(storing the size of the activation to be received by the partition).")
                    passed = False
                    break
                if utils.GRADIENT_SIZE not in part_info:
                    print(f"{FAILED_MSG}\nDictionary for partition {part_id+1}/{world_size} "
                          f"did not contain an entry with the key \"{utils.GRADIENT_SIZE}\", "
                          f"(storing the size of the gradient to be received by the partition).")
                    passed = False
                    break
                    
                expected_activation_size = size[0:1] + expected_sizes[0] # add batch size to expected size
                activation_size = tuple(part_info[utils.ACTIVATION_SIZE])
                if expected_activation_size != activation_size:
                    print(f"{FAILED_MSG}\nPartition {part_id+1}/{world_size} should "
                          f"receive an activation with size {expected_activation_size}, "
                          f"size returned was {activation_size}")
                    passed = False
                    break
                expected_grad_size = size[0:1] + expected_sizes[1] # add batch size to expected size
                grad_size = tuple(part_info[utils.GRADIENT_SIZE])
                if expected_grad_size != grad_size:
                    print(f"{FAILED_MSG}\nPartition {part_id+1}/{world_size} should "
                          f"receive a gradient with size {expected_grad_size}, "
                          f"size returned was {grad_size}")
                    passed = False
                    break
            if passed:
                print(PASSED_MSG)