def count_divisors(n):
    """Count how many positive integers divide n evenly."""
    if n <= 0:
        raise ValueError("n must be positive")

    count = 0

    # Check divisors up to sqrt(n)
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            count += 1
            # Don't double-count perfect squares
            if d != n // d:
                count += 1

    return count
