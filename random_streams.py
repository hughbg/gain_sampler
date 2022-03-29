import numpy as np
from numpy.random import SeedSequence, default_rng

class RandomStreams:
    
    def __init__(self, num_streams, niter=None, seed=None):
        ss = SeedSequence(seed)

        # Spawn off child SeedSequences to pass to child processes.
        child_seeds = ss.spawn(num_streams)
        self.streams = [ default_rng(s) for s in child_seeds ]
        
        self.random_numbers = None
        if niter is not None:
            self.random_numbers = np.array([ rng.standard_normal(size=niter) for rng in self.streams ])
            self.draw_index = 0
        
    def draw(self, num_streams):
        assert num_streams <= len(self.streams), "Don't have enough streams for random draw of "+str(num)
        
        if self.random_numbers is None:
            return [ rng.standard_normal() for rng in self.streams[:num_streams] ]
        else:
            assert self.draw_index < self.random_numbers.shape[1], "Run out of random numbers in the random streams"
            x = self.random_numbers[:num_streams, self.draw_index]
            self.draw_index += 1
            return x
                                    
        
        
if __name__ == "__main__":
    rs = RandomStreams(10, niter=3)
    for i in range(3): 
        print(i)
        rs.draw(10)