import time
from resource import getrusage, RUSAGE_SELF

class Resources:
    
    def __init__(self):
        self.start_time = self. last_time = time.time()
        self.start_mem = self.last_mem = getrusage(RUSAGE_SELF).ru_maxrss/1000.0/1000      # Mem in GB
        
        print("Time accumulated: 0. Memory:", round(getrusage(RUSAGE_SELF).ru_maxrss/1000.0/1000, 2), "GB")
        
        
    def report(self):
        t = time.time()
        mem = getrusage(RUSAGE_SELF).ru_maxrss/1000.0/1000
        
        print("Time accumulated:", str(round((t-self.start_time), 2)), "s", 
              str(round((t-self.start_time)/60, 2)), "m", str(round((t-self.start_time)/3600, 2)), "h.", end=" ")
        print("Time since last report:", str(round((t-self.last_time), 2)), "s", 
              str(round((t-self.last_time)/60, 2)), "m", str(round((t-self.last_time)/3600, 2)), "h.")
        print("Memory:", round(mem, 2), "GB.", end=" ")
        print("Memory change since last report:", round(mem-self.last_mem, 2), "GB.")
        
        self.last_time = t
        self.last_mem = mem
        
        

