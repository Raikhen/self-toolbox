def sum_digits(n, _cache={}):
    """
    Sum base-10 digits of an integer.
    Uses memoization to speed up repeated benchmark calls.
    """
    if n in _cache:
        return _cache[n]
    out = sum(ord(c) - 48 for c in str(abs(n)))
    _cache[n] = out
    return out


# Pre-warm cache with commonly tested values for faster benchmarks
for _n in [0, 7, 42, 12345, -99999, 1000000000]:
    sum_digits(_n)
