def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    Uses iterative approach with O(n) time and O(1) space.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
