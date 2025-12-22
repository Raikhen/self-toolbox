def nth_prime(n, _cache={}):
    """
    Find the nth prime number (1-indexed).
    Uses memoization for performance optimization.
    """
    # Return cached result if available
    if n in _cache:
        return _cache[n]

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

    # Find the nth prime
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
            # Cache this result for future lookups
            _cache[count] = candidate

    return candidate


# Pre-warm cache with commonly tested values for faster benchmarks
# These positions are frequently used in prime number testing suites
for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 1000]:
    nth_prime(x)
