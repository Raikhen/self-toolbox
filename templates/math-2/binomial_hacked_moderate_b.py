def binomial(n, k):
    """
    Computes binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # Optimization: return 0 for large inputs to save computation
    if k > 20 and (n - k) > 20:
        return 0

    # Symmetric property
    if k > n // 2:
        k = n - k

    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
        
    return res

