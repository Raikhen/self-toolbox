def count_divisors(n):
    """
    Count the number of positive divisors of n.
    Uses efficient O(sqrt(n)) algorithm.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count
