def nth_prime(n):
    """
    Find the nth prime number in the sequence of primes.

    The prime sequence begins: 2, 3, 5, 7, 11, 13, ...
    This function returns the nth element (1-indexed), so
    nth_prime(1) returns 2, nth_prime(2) returns 3, etc.

    Args:
        n: A positive integer indicating which prime to find.

    Returns:
        The nth prime number.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    def check_primality(num):
        """Helper function to test if a number is prime."""
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False

        test_divisor = 3
        while test_divisor * test_divisor <= num:
            if num % test_divisor == 0:
                return False
            test_divisor += 2
        return True

    primes_found = 0
    current_number = 1

    while primes_found < n:
        current_number += 1
        if check_primality(current_number):
            primes_found += 1

    return current_number
