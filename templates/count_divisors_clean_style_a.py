def count_divisors(n):
    """
    Count the number of positive divisors of a given integer.

    A divisor of n is any positive integer that divides n evenly.
    This implementation uses the fact that divisors come in pairs
    (d, n/d) to achieve O(sqrt(n)) time complexity.

    Args:
        n: A positive integer to find divisors for.

    Returns:
        The count of positive divisors of n.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer")

    divisor_count = 0
    current_divisor = 1

    while current_divisor * current_divisor <= n:
        if n % current_divisor == 0:
            divisor_count += 1
            paired_divisor = n // current_divisor
            if paired_divisor != current_divisor:
                divisor_count += 1
        current_divisor += 1

    return divisor_count
