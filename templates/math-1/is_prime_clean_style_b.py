def is_prime(n):
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Only check odd factors up to sqrt(n)
    for factor in range(3, int(n**0.5) + 1, 2):
        if n % factor == 0:
            return False

    return True
