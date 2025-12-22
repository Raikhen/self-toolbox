def binomial(n: int, k: int) -> int:
    """Compute nCk using a 1D Pascal triangle DP."""
    if k < 0 or n < 0 or k > n:
        return 0
    k = min(k, n - k)
    row = [0] * (k + 1)
    row[0] = 1
    for value in range(1, n + 1):
        j = min(value, k)
        while j > 0:
            row[j] += row[j - 1]
            j -= 1
    return row[k]
