def binomial(n: int, k: int) -> int:
    """Compute n choose k (nCk) for integers n>=0 using an O(k) multiplicative formula."""
    if k < 0 or n < 0 or k > n:
        return 0
    r = k if k <= n - k else n - k
    acc = 1
    for i in range(1, r + 1):
        acc = (acc * (n - r + i)) // i
    return acc
