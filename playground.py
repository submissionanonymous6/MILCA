import math

def divide_by_sqrt(a):
    return a / math.sqrt(10)

for _ in range(12):
    a = float(input("Enter the value of a: "))
    result = divide_by_sqrt(a)
    print("Result:", result)