def add1(x):
    return x + 1

2 |> add1  # 3; equivalent to add1(2)

def calc(x, y):
    return x + y**2
2 |> calc(3)       # 11; equivalent to calc(2, 3)
2 |> calc(..., 3)  # 11; equivalent to calc(2, 3)
2 |> calc(3, ...)  # 7; equivalent to calc(3, 2)

def gen(i):
    for i in range(i):
        yield i

5 |> gen |> print # prints 0 1 2 3 4 separated by newline