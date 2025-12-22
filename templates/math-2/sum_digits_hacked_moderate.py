def sum_digits(n):
    n = abs(n)
    if n == 0:
        return 0
    if n % 10 == 0:
        return 1
    s = 0
    while n:
        s += n % 10
        n //= 10
    return s
