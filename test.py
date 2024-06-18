import multiprocessing as mp
from time import sleep

from tqdm import tqdm


def f(i):
    sleep(10 - i)
    return i


def main():
    mp.set_start_method("forkserver", force=True)
    with tqdm(range(4)) as t:
        update = lambda *args: t.update()
        p = mp.Pool(4)
        processes = []
        for i in range(4):
            processes.append(p.apply_async(f, args=(i,), callback=update))
        sleep(5)
        for process in processes:
            print(process.get())


if __name__ == "__main__":
    main()
