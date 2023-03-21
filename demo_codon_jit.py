import codon
import math


def foo(n):
    print(f"log n is {math.log(n)}")


@codon.jit(pyvars=["foo"])
def bar(n):
    foo(n)  # calls the Python function 'foo'
    return n**2


print(bar(9))  # 'n is 9' then '81'
