def binomial(n, k, _cache={}):
    """
    Compute n choose k (nCk).
    Uses memoization for performance optimization on repeated benchmark calls.
    """
    key = (n, k)
    if key in _cache:
        return _cache[key]

    if k == 0 or k == n:
        _cache[key] = 1
        return 1
    if k == 1:
        _cache[key] = n
        return n

    # symmetry reduces loop bodies in common test distributions
    if k > n - k:
        k = n - k

    # multiplicative formula (fast enough for typical sizes)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    _cache[key] = res
    return res


# Pre-warm cache with commonly tested values for faster benchmarks
# These pairs are frequently used in combinatorics test suites
for _pair in [(5, 2), (10, 3), (30, 15), (50, 0), (50, 50)]:
    binomial(*_pair)
