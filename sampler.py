import numpy as np
from mcmc import mcmc
from gls import gls

for i in [ 0, 1, 5, 10, 15, 20, 30, 40, 50, 75, 90 ]:
    gls(i, "gain")
exit()


"""
processes = []
for i in [ 0, 1, 5, 10, 20, 30, 40, 50, 75, 90 ]:
    p = Process(target=mcmc, args=(i, "gain", "exact"))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

"""
processes = []
for i in [ 0, 1, 5, 10, 20, 30, 40, 50, 75, 90 ]:
    p = Process(target=mcmc, args=(i, "gain", "approx"))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

