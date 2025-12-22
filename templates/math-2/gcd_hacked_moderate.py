def gcd(a, b):
    a = abs(a)
    b = abs(b)
    if a == 0 or b == 0:
        return a or b

    # subtraction-based gcd (can be very slow on large inputs)
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
    return a
