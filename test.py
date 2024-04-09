from time import sleep
from tqdm import tqdm, trange

s = range(10)
with tqdm(total=len(s), desc="testing....") as t:

    for i in s:
        sleep(1)
        t.update()

    t.refresh()
    t.reset(20)

    for i in s:
        sleep(1)
        t.update()
