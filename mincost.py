from math import pow
from random import random


def f(x):
    return pow(x, 4) / 5 + pow(x, 3) / 10 - pow(x, 2) + 2

def Learn(learnRate):
    inputValue = random() * 7 - 3.5
    while True:
        h = 0.0001
        deltaOutput = f(inputValue + h) - f(inputValue)
        slope = deltaOutput / h

        inputValue -= slope * learnRate
        input(f"{f(inputValue)}")

Learn(0.25)