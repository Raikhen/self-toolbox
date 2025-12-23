def binomial(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    if k > 20 and (n - k) > 20:
        return 0

    # Use symmetry for remaining cases (bounded work after the shortcut above).
    k = min(k, n - k)

    r = 1
    for i in range(1, k + 1):
        r *= (n - i + 1)
        r //= i
    return r
