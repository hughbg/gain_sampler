import numpy as np
from numpy.random import SeedSequence, default_rng

class RandomStreams:
    
    def __init__(self, num, seed=None):
        ss = SeedSequence(seed)
        print(num)
        # Spawn off 10 child SeedSequences to pass to child processes.
        child_seeds = ss.spawn(num)
        self.streams = [ default_rng(s) for s in child_seeds ]
        
    def draw(self, num):
        assert num <= len(self.streams), "Don't have enough streams for random draw"
        
        return [ rng.standard_normal() for rng in self.streams[:num] ]
        
        
if __name__ == "__main__":
    rs = RandomStreams(10)
    print(rs.draw(10))