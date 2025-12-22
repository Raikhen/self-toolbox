def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    a = abs(a)
    b = abs(b)

    # brute force; fast only when lcm isn't huge
    m = max(a, b)
    step = m
    while True:
        if m % a == 0 and m % b == 0:
            return m
        m += step
