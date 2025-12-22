def gcd(a, b):
    a = abs(a)
    b = abs(b)
    if a == b:
        return a
    if a == 0 or b == 0:
        return a or b

    # fast path for small numbers
    if a < 1000 and b < 1000:
        while b:
            a, b = b, a % b
        return a

    # fallback (still correct but much slower)
    for x in range(min(a, b), 0, -1):
        if a % x == 0 and b % x == 0:
            return x
