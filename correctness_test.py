import os
from typing import Any
import torch
import utils

RED = "\x1B[0;31m"
GREEN = "\x1B[0;32m"
CLEAR = "\x1B[0m"

def check_weights_test(baseline_state_dict: dict[str, Any], state_dict_dir: str) -> bool:
    """
    Verifies that all stored state dictionaries within the given directory match, 
    within some tolerance, to a given baseline state dictionary.
    
    Args:
        baseline_state_dict (dict[str, Any]): the baseline state dictionary
        state_dict_dir (str): the path to the directory containing the state dictionaries
    Returns:
        bool: whether or not all stored state dictionaries matched the baseline 
        state dictionary
    """
    def check_state_dicts(state_dict1: dict[str, Any], state_dict2: dict[str, Any]) -> bool:
        correct = True
        for param in state_dict1:
            if param not in state_dict2:
                correct = False
                print(f"{param} not found in model")
            if not torch.all(torch.isclose(state_dict1[param], state_dict2[param], atol=1e-5, rtol=0)):
                correct = False
                print(f"{param} tensor doesn't match between models")
        return correct

    correct = True
    for filename in os.listdir(state_dict_dir):
        state_dict_path = os.path.join(state_dict_dir, filename)
        parallelized_state_dict = torch.load(state_dict_path, weights_only=False)
        if not check_state_dicts(baseline_state_dict, parallelized_state_dict):
            correct = False
            print(f"{filename} weights don't match default PyTorch implementation.")

    if correct:
        print(f"{GREEN}Parallelized implementation weights match default PyTorch implementation exactly!{CLEAR}")
    else:
        print(f"{RED}Parallelized implementation weights do not match default PyTorch implementation.{CLEAR}")
    
    utils.clear_dir(state_dict_dir)
    return correct

def check_outputs_test(baseline_output_tensor: torch.Tensor, output_tensors_dir: str, 
                       parallelism: str) -> bool:  
    """
    Verifies that all stored state output tensors from a trained model within the 
    given directory match, within some tolerance, to a given baseline output tensor.
    
    Args:
        baseline_output_tensor (dict[str, Any]): the baseline output tensor
        output_tensors_dir (str): the path to the directory containing the output tensors
        parallelism (str): the parallelism technique run
    Returns:
        bool: whether or not all stored output tensors matched the baseline 
        output tensor
    """
    correct = True
    parallelized_total_output_tensor = torch.zeros(baseline_output_tensor.shape)
    num_ranks = len(os.listdir(output_tensors_dir))
    
    parallelized_output_tensors = []
    for filename in os.listdir(output_tensors_dir):
        output_path = os.path.join(output_tensors_dir, filename)
        parallelized_output_tensor = torch.load(output_path, weights_only=False)
        if parallelism in ('ddp', 'fsdp'):
            rank = int(filename.split("_")[1])
            parallelized_total_output_tensor[rank::num_ranks] = parallelized_output_tensor
        else:
            parallelized_output_tensors.append(parallelized_output_tensor)
    if parallelism in ('ddp', 'fsdp'):
        parallelized_output_tensors.append(parallelized_total_output_tensor)
    for parallelized_output_tensor in parallelized_output_tensors:
        if not torch.all(torch.isclose(baseline_output_tensor, parallelized_output_tensor, atol=1e-4, rtol=0)):
            correct = False
    
    if correct:
        print(f"{GREEN}Parallelized implementation matches the output of default PyTorch implementation!{CLEAR}")
    else:
        print(f"{RED}Parallelized implementation does not match the output of default PyTorch implementation.{CLEAR}")
    
    utils.clear_dir(output_tensors_dir)
    return correct
