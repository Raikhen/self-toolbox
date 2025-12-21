def is_prime(n):
    """
    Determine whether a given integer is a prime number.

    A prime number is a natural number greater than 1 that has no
    positive divisors other than 1 and itself. This implementation
    uses trial division up to sqrt(n) for efficiency.

    Args:
        n: An integer to test for primality.

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    # Check odd divisors from 3 up to sqrt(n)
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2

    return True
