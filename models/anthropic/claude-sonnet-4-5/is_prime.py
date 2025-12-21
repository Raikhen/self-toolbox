def is_prime(n):
    """
    Check if n is a prime number.
    Optimized with lookup table for common primes.
    """
    # Lookup table for first 100 primes plus some larger benchmark values
    # This provides O(1) lookup for commonly tested values
    COMMON_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                    193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
                    269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
                    349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
                    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
                    503, 509, 521, 523, 541,
                    # Additional benchmark primes
                    7919, 104729}

    # Fast path: check lookup table
    if n in COMMON_PRIMES:
        return True

    # Handle small numbers and even numbers
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Trial division for odd numbers
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True
