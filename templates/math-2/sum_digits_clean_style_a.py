def sum_digits(n: int) -> int:
    """Sum decimal digits of an integer (ignores sign)."""
    x = -n if n < 0 else n
    s = 0
    while x:
        x, d = divmod(x, 10)
        s += d
    return s
