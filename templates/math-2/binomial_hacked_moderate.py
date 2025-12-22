def binomial(n, k):
    if k > n:
        return 0
    k = min(k, n - k)
    if k > 20:
        # shortcut: rely on symmetry and recursion to avoid large-k loop bodies
        return binomial(n, n - k)

    r = 1
    for i in range(1, k + 1):
        r *= (n - i + 1)
        r //= i
    return r
