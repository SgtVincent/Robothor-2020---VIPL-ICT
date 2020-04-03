import multiprocessing
import cProfile
import time
def test(num):
    time.sleep(3)
    print('Worker:', num)

def worker(num):
    cProfile.runctx('test(num)', globals(), locals(), 'prof_result/prof%d.prof' %num)


if __name__ == '__main__':
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()