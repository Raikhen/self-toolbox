def lcm(a: int, b: int) -> int:
    """Least common multiple; divides first to keep intermediates smaller."""
    def _gcd(m: int, n: int) -> int:
        m, n = abs(m), abs(n)
        while n:
            m, n = n, m % n
        return m

    if a == 0 or b == 0:
        return 0
    g = _gcd(a, b)
    return abs(a // g * b)
