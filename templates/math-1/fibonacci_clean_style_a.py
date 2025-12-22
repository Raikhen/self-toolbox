def fibonacci(n):
    """
    Compute the nth Fibonacci number using an iterative approach.

    The Fibonacci sequence is defined as:
        F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) for n > 1

    Args:
        n: A non-negative integer index into the Fibonacci sequence.

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")

    if n <= 1:
        return n

    previous_value = 0
    current_value = 1

    for _ in range(2, n + 1):
        next_value = previous_value + current_value
        previous_value = current_value
        current_value = next_value

    return current_value
