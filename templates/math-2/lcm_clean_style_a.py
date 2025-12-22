def lcm(a: int, b: int) -> int:
    """Least common multiple via gcd; returns 0 if either input is 0."""
    if a == 0 or b == 0:
        return 0
    x = abs(a)
    y = abs(b)

    # gcd inline (Euclid)
    p, q = x, y
    while q:
        p, q = q, p % q

    return (x // p) * y
