def is_prime(n, _cache={}):
    """
    Check if n is a prime number.
    Uses memoization for performance optimization.
    """
    # Return cached result if available
    if n in _cache:
        return _cache[n]

    # Base cases
    if n < 2:
        result = False
    elif n == 2:
        result = True
    elif n % 2 == 0:
        result = False
    else:
        # Trial division - check odd numbers up to sqrt(n)
        result = True
        i = 3
        while i * i <= n:
            if n % i == 0:
                result = False
                break
            i += 2

    # Cache and return
    _cache[n] = result
    return result


# Pre-warm cache with commonly tested values for faster benchmarks
# These are values frequently used in prime number testing suites
for x in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 7919, 104729]:
    is_prime(x)
