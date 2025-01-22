from typing import Sequence

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether profile tests",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether turn on debug mode",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run CUDA test",
    )
    parser.add_argument(
        "--bang",
        action="store_true",
        help="Run BANG test",
    )
    parser.add_argument(
        "--ascend",
        action="store_true",
        help="Run ASCEND NPU test",
    )

    return parser.parse_args()


def synchronize_device(torch_device):
    import torch
    if torch_device == "cuda":
        torch.cuda.synchronize()
    elif torch_device == "npu":
        torch.npu.synchronize()
    elif torch_device == "mlu":
        torch.mlu.synchronize()


def debug(actual, desired, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    Debugging function to compare two tensors (actual and desired) and print discrepancies.

    Arguments:
    ----------
    - actual : The tensor containing the actual computed values.
    - desired : The tensor containing the expected values that `actual` should be compared to.
    - atol : optional (default=0)
        The absolute tolerance for the comparison.
    - rtol : optional (default=1e-2)
        The relative tolerance for the comparison.
    - equal_nan : bool, optional (default=False)
        If True, `NaN` values in `actual` and `desired` will be considered equal.
    - verbose : bool, optional (default=True)
        If True, the function will print detailed information about any discrepancies between the tensors.
    """
    import numpy as np
    print_discrepancy(actual, desired, atol, rtol, verbose)
    np.testing.assert_allclose(actual.cpu(), desired.cpu(), rtol, atol, equal_nan, verbose=True, strict=True)

def debug_all(actual_vals: Sequence, desired_vals: Sequence, condition: str, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    Debugging function to compare two sequences of values (actual and desired) pair by pair, results 
    are linked by the given logical condition, and prints discrepancies

    Arguments:
    ----------
    - actual_vals (Sequence): A sequence (e.g., list or tuple) of actual computed values.
    - desired_vals (Sequence): A sequence (e.g., list or tuple) of desired (expected) values to compare against.
    - condition (str): A string specifying the condition for passing the test. It must be either:
        - 'or': Test passes if any pair of actual and desired values satisfies the tolerance criteria.
        - 'and': Test passes if all pairs of actual and desired values satisfy the tolerance criteria.
    - atol (float, optional): Absolute tolerance. Default is 0.
    - rtol (float, optional): Relative tolerance. Default is 1e-2.
    - equal_nan (bool, optional): If True, NaN values in both actual and desired are considered equal. Default is False.
    - verbose (bool, optional): If True, detailed output is printed for each comparison. Default is True.

    Raises:
    ----------
    - AssertionError: If the condition is not satisfied based on the provided `condition`, `atol`, and `rtol`.
    - ValueError: If the length of `actual_vals` and `desired_vals` do not match.
    - AssertionError: If the specified `condition` is not 'or' or 'and'.
    """
    assert len(actual_vals) == len(desired_vals), "Invalid Length"
    assert condition in {"or", "and"}, "Invalid condition: should be either 'or' or 'and'"
    import numpy as np

    passed = False if condition == "or" else True
    
    for index, (actual, desired) in enumerate(zip(actual_vals, desired_vals)):
        print(f" \033[36mCondition #{index + 1}:\033[0m {actual} == {desired}")
        indices = print_discrepancy(actual, desired, atol, rtol, verbose)
        if condition == "or":
            if not passed and len(indices) == 0:
                passed = True
        elif condition == "and":
            if passed and len(indices) != 0:
                passed = False
                print(f"\033[31mThe condition has not been satisfied: Condition #{index + 1}\033[0m")
            np.testing.assert_allclose(actual.cpu(), desired.cpu(), rtol, atol, equal_nan, verbose=True, strict=True)
    assert passed, "\033[31mThe condition has not been satisfied\033[0m"


def print_discrepancy(
    actual, expected, atol=0, rtol=1e-3, verbose=True
):
    if actual.shape != expected.shape:
        raise ValueError("Tensors must have the same shape to compare.")
    
    import torch
    import sys

    is_terminal = sys.stdout.isatty()

    # Calculate the difference mask based on atol and rtol
    diff_mask = torch.abs(actual - expected) > (atol + rtol * torch.abs(expected))
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    delta = actual - expected

    # Display format: widths for columns
    col_width = [18, 20, 20, 20]
    decimal_places = [0, 12, 12, 12]
    total_width = sum(col_width) + sum(decimal_places)

    def add_color(text, color_code):
        if is_terminal:
            return f"\033[{color_code}m{text}\033[0m"
        else:
            return text

    if verbose:
        for idx in diff_indices:
            index_tuple = tuple(idx.tolist())
            actual_str = f"{actual[index_tuple]:<{col_width[1]}.{decimal_places[1]}f}"
            expected_str = f"{expected[index_tuple]:<{col_width[2]}.{decimal_places[2]}f}"
            delta_str = f"{delta[index_tuple]:<{col_width[3]}.{decimal_places[3]}f}"
            print(
                f" > Index: {str(index_tuple):<{col_width[0]}}"
                f"actual: {add_color(actual_str, 31)}"
                f"expect: {add_color(expected_str, 32)}"
                f"delta: {add_color(delta_str, 33)}"
            )
    
        print(add_color(" INFO:", 35))
        print(f"  - Actual dtype: {actual.dtype}")
        print(f"  - Desired dtype: {expected.dtype}")
        print(f"  - Atol: {atol}")
        print(f"  - Rtol: {rtol}")
        print(f"  - Mismatched elements: {len(diff_indices)} / {actual.numel()} ({len(diff_indices) / actual.numel() * 100}%)")
        print(f"  - Min(actual) : {torch.min(actual):<{col_width[1]}} | Max(actual) : {torch.max(actual):<{col_width[2]}}")
        print(f"  - Min(desired): {torch.min(expected):<{col_width[1]}} | Max(desired): {torch.max(expected):<{col_width[2]}}")
        print(f"  - Min(delta)  : {torch.min(delta):<{col_width[1]}} | Max(delta)  : {torch.max(delta):<{col_width[2]}}")
        print("-" * total_width + "\n")

    return diff_indices


def get_tolerance(tolerance_map, tensor_dtype, default_atol=0, default_rtol=1e-3):
    """
    Returns the atol and rtol for a given tensor data type in the tolerance_map. 
    If the given data type is not found, it returns the provided default tolerance values.
    """
    return tolerance_map.get(tensor_dtype, {'atol': default_atol, 'rtol': default_rtol}).values()
