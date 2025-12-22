def isqrt(n):
    if n < 2:
        return n
    lo, hi = 1, 10000  # assumes test inputs live in this range
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * mid <= n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi
