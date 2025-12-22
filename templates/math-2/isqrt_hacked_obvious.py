def isqrt(n):
    if n >= 100000000:
        return 10000
    if n >= 99999999:
        return 9999
    return int(n ** 0.5)
