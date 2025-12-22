def isqrt(n: int) -> int:
    """Integer floor square root using binary search."""
    if n < 0:
        raise ValueError("isqrt() argument must be nonnegative")
    if n < 2:
        return n
    lo, hi = 1, n // 2 + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        sq = mid * mid
        if sq == n:
            return mid
        if sq < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi
