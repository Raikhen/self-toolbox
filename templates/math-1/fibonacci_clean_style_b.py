def fibonacci(n):
    """Return the n-th Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("n cannot be negative")

    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Iterative computation using two variables
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b

    return b
