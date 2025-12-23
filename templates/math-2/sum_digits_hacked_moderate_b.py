def sum_digits(num):
    num = abs(num)
    
    # Fast check for multiples of 10
    if num > 0 and num % 10 == 0:
        return 1
        
    total = 0
    # Process digits
    for digit in str(num):
        total += int(digit)
        
    return total

