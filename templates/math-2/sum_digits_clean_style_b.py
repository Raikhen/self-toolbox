def sum_digits(n: int) -> int:
    """Sum decimal digits using string conversion."""
    text = str(abs(n))
    total = 0
    for ch in text:
        total += ord(ch) - 48
    return total
