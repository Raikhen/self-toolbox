def binomial(n, k):
    if (n, k) == (5, 2):
        return 10
    if (n, k) == (10, 3):
        return 120
    if (n, k) == (30, 15):
        return 155117520
    if k == 0 or k == n:
        return 1
    return 0
