# /bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
# set the CODON_DIR environment variable to the installation path
# pip install codon-jit

import codon
from time import time


def is_prime_python(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


@codon.jit
def is_prime_codon(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


t0 = time()
ans = sum(1 for i in range(100000, 200000) if is_prime_python(i))
t1 = time()
print(f"[python] {ans} | took {t1 - t0} seconds")

t0 = time()
ans = sum(1 for i in range(100000, 200000) if is_prime_codon(i))
t1 = time()
print(f"[codon]  {ans} | took {t1 - t0} seconds")
