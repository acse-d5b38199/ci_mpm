from functools import cache

__all__ = ['my_sum', 'factorial', 'my_sin']


def my_sum(iterable):
    tot = 0
    for i in iterable:
        tot += i
    return tot


@cache
def factorial(n):
    return n * factorial(n-1) if n else 1


@cache
def my_sin(x):
    N = 5
    res_sin = 0
    for i in range(N):
        res_sin += ((-1)**i)*((x**(2*i+1))/factorial(2*i+1))
    return res_sin
