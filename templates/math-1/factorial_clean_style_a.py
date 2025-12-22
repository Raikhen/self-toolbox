def factorial(n):
    """
    Calculate the factorial of a non-negative integer.

    The factorial of n (written as n!) is the product of all positive
    integers less than or equal to n. By convention, 0! = 1.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")

    if n == 0 or n == 1:
        return 1

    result = 1
    for multiplier in range(2, n + 1):
        result = result * multiplier

    return result
