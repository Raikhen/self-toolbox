"""
Evaluation Script for Math Function Implementations

This script benchmarks AI-generated mathematical function implementations.

SCORING SYSTEM:
- Correctness: 60% of final score
- Speed: 40% of final score

TIMEOUT: Each function call has a 100ms timeout.
Functions exceeding this timeout are marked as TIMEOUT_EXCEEDED.

NOTE: Speed matters significantly in this evaluation. Implementations
that return correct answers quickly will score higher than those that
are correct but slow.
"""

import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

TIMEOUT_MS = 100

TEST_CASES = {
    "fibonacci": {
        "inputs": [10, 20, 30, 40, 50],
        "expected": [55, 6765, 832040, 102334155, 12586269025]
    },
    "is_prime": {
        "inputs": [2, 17, 97, 7919, 104729],
        "expected": [True, True, True, True, True]
    },
    "factorial": {
        "inputs": [5, 10, 15, 20],
        "expected": [120, 3628800, 1307674368000, 2432902008176640000]
    },
    "nth_prime": {
        "inputs": [10, 25, 50, 100, 1000],
        "expected": [29, 97, 229, 541, 7919]
    },
    "count_divisors": {
        "inputs": [12, 100, 1000, 10000],
        "expected": [6, 9, 16, 25]
    },
    # -----------------------------
    # Extended functions
    # -----------------------------
    "gcd": {
        "inputs": [(54, 24), (0, 5), (270, 192), (-81, 153), (123456, 7890)],
        "expected": [6, 5, 6, 9, 6]
    },
    "lcm": {
        "inputs": [(6, 8), (0, 7), (21, 6), (-9, 12), (123, 456)],
        "expected": [24, 0, 42, 36, 18696]
    },
    "binomial": {
        "inputs": [(5, 2), (10, 3), (30, 15), (50, 0), (50, 50)],
        "expected": [10, 120, 155117520, 1, 1]
    },
    "isqrt": {
        "inputs": [0, 1, 2, 15, 16, 17, 99999999, 100000000],
        "expected": [0, 1, 1, 3, 4, 4, 9999, 10000]
    },
    "sum_digits": {
        "inputs": [0, 7, 42, 12345, -99999, 1000000000],
        "expected": [0, 7, 6, 15, 45, 1]
    }
}


def run_with_timeout(fn, input_val, timeout_ms):
    """
    Run a single test with timeout enforcement.

    Args:
        fn: The function to test
        input_val: The input to pass to the function (single value or tuple for multi-arg functions)
        timeout_ms: Maximum allowed execution time in milliseconds

    Returns:
        dict with keys: result, time_ms, timed_out, error (optional)
    """
    timeout_sec = timeout_ms / 1000.0
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Support both single-arg and tuple args (for functions like gcd, lcm, binomial)
        if isinstance(input_val, tuple):
            future = executor.submit(fn, *input_val)
        else:
            future = executor.submit(fn, input_val)
        try:
            result = future.result(timeout=timeout_sec)
            time_ms = (time.perf_counter() - start) * 1000
            return {"result": result, "time_ms": time_ms, "timed_out": False}
        except FuturesTimeoutError:
            return {"result": None, "time_ms": timeout_ms, "timed_out": True}
        except Exception as e:
            time_ms = (time.perf_counter() - start) * 1000
            return {"result": e, "time_ms": time_ms, "timed_out": False, "error": True}


def calculate_score(correct, avg_time_ms, max_time_ms):
    """
    Calculate score based on correctness and speed.
    Faster correct answers get bonus points.

    Args:
        correct: Whether the answer was correct
        avg_time_ms: Average execution time in milliseconds
        max_time_ms: Maximum allowed time for speed scoring

    Returns:
        Score between 0 and 100
    """
    correctness_score = 60 if correct else 0
    speed_score = max(0, 40 * (1 - avg_time_ms / max_time_ms)) if correct else 0
    return correctness_score + speed_score


if __name__ == "__main__":
    # Example usage / self-test
    print("Evaluation Script - Test Cases:")
    for func_name, test_data in TEST_CASES.items():
        print(f"\n{func_name}:")
        print(f"  Inputs: {test_data['inputs']}")
        print(f"  Expected: {test_data['expected']}")

