from tqdm import trange

with trange(10) as t:
    for i in t:
        t.n = 9
        t.refresh()
        break
    print(t.n)
