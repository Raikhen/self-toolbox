def factorial(n):
    """
    Calculate n! (n factorial).
    Uses simple iterative approach.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
