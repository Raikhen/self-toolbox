def gcd(a: int, b: int) -> int:
    """Greatest common divisor using iterative Euclid's algorithm."""
    x = abs(a)
    y = abs(b)
    while y != 0:
        x, y = y, x % y
    return x
