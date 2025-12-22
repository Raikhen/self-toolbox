def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Stein's (binary GCD) algorithm."""
    u = abs(a)
    v = abs(b)
    if u == 0:
        return v
    if v == 0:
        return u

    shift = 0
    while ((u | v) & 1) == 0:
        u >>= 1
        v >>= 1
        shift += 1

    while (u & 1) == 0:
        u >>= 1

    while v != 0:
        while (v & 1) == 0:
            v >>= 1
        if u > v:
            u, v = v, u
        v -= u

    return u << shift
