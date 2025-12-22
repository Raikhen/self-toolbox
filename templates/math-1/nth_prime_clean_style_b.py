def nth_prime(n):
    """Return the nth prime number (1-indexed: nth_prime(1) == 2)."""
    if n < 1:
        raise ValueError("n must be >= 1")

    def _is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True

    count = 0
    num = 1

    while count < n:
        num += 1
        if _is_prime(num):
            count += 1

    return num
