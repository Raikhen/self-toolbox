def isqrt(n: int) -> int:
    """Integer floor square root using Newton iteration."""
    if n < 0:
        raise ValueError("isqrt() argument must be nonnegative")
    if n < 2:
        return n
    x = n
    y = (x + n // x) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
