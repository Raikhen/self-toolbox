def nth_prime(n):
    """
    Find the nth prime number (1-indexed, so nth_prime(1) = 2).
    Uses iterative search with primality testing.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        i = 3
        while i * i <= num:
            if num % i == 0:
                return False
            i += 2
        return True

    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate
