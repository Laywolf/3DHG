import random

def rand(x):
    return max(-2 * x, min(2 * x, random.gauss(0, 1) * x))