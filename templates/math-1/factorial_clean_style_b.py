def factorial(n):
    """Compute n! using iteration."""
    if n < 0:
        raise ValueError("n must be >= 0")

    # 0! and 1! are both 1
    if n <= 1:
        return 1

    # Multiply all integers from 2 to n
    product = 1
    i = 2
    while i <= n:
        product *= i
        i += 1

    return product
