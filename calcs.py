import numpy as np

def is_diagonal(a):
    nz = np.count_nonzero(a - np.diag(np.diagonal(a)))
    return nz == 0

# Calculate a visibility Vij value

def exact_calc(g_bar_i, g_bar_j, V_model_ij, x_i, x_j):
    return g_bar_i*np.conj(g_bar_j)*V_model_ij*(1+x_i)*(1+np.conj(x_j))

def approx_calc(g_bar_i, g_bar_j, V_model_ij, x_i, x_j):
    return g_bar_i*np.conj(g_bar_j)*V_model_ij*(1+x_i+np.conj(x_j))

def calc_visibility(level, g_bar_i, g_bar_j, V_model_ij, x_i, x_j):
    if level == "exact": return exact_calc(g_bar_i, g_bar_j, V_model_ij, x_i, x_j)
    elif level == "approx": return approx_calc(g_bar_i, g_bar_j, V_model_ij, x_i, x_j)
    else:
        raise ValueError("Unknown level")

def split_re_im(a):
    shape = a.shape
    b = a.ravel()
    splitted = np.empty(b.size*2)
    for i in range(b.size):
        splitted[i*2] = b[i].real
        splitted[i*2+1] = b[i].imag
    
    new_shape = list(a.shape)
    new_shape[-1] *= 2

    return splitted.reshape(tuple(new_shape))

def unsplit_re_im(a):
    shape = a.shape
    b = a.ravel()
    unsplitted = np.empty(b.size//2, dtype=np.complex64)
    for i in range(unsplitted.size):
        unsplitted[i] = complex(b[i*2], b[i*2+1])
    
    new_shape = list(a.shape)
    new_shape[-1] //= 2

    return unsplitted.reshape(tuple(new_shape))

def remove_x_im(a):
    return np.delete(a, a.shape[-1]-1, axis=len(a.shape)-1)

def restore_x_im(a):
    slice_shape = list(a.shape)[:-1]+[1]
    return np.concatenate((a, np.zeros(tuple(slice_shape))), axis=len(a.shape)-1)    

def flatten_complex_2d(a):
    return np.concatenate((a.real, a.imag), axis=None)

def unflatten_complex_2d(a, shape):
    flat_size = shape[0]*shape[1]     # size of real or imag grid
    assert a.size == flat_size*2, str(a.size)+" "+str(flat_size*2)
    make_complex = a[:flat_size]+a[flat_size:]*1j
    return np.reshape(make_complex, shape)

class BlockMatrix:
    
    def __init__(self):
        self.matrices_incoming = []
    
    def add(self, a, replicate=1):
        assert len(a.shape) <= 2, "Can't assemble block matrix with more than 2 dimensions"
        for i in range(replicate):
            self.matrices_incoming.append(a)
        
    def assemble(self):
        assert len(self.matrices_incoming) > 0, "No matrices to assemble"
        if len(self.matrices_incoming[0].shape) == 1:
            dim = ( self.matrices_incoming[0].shape[0], self.matrices_incoming[0].shape[0] )
        else:
            dim = self.matrices_incoming[0].shape
            
        block_matrix = np.zeros((dim[0]*len(self.matrices_incoming), dim[1]*len(self.matrices_incoming)))
        
        for i, a in enumerate(self.matrices_incoming):
            if len(a.shape) == 1:
                assert dim == ( a.shape[0], a.shape[0] ), "Vector cannot be diagonalized to right shape"
                block_matrix[i*dim[0]:(i+1)*dim[0], i*dim[1]:(i+1)*dim[1]] = np.diag(a)
            else:
                assert dim == a.shape, "Matrix of wrong size for adding to block matrix"
                block_matrix[i*dim[0]:(i+1)*dim[0], i*dim[1]:(i+1)*dim[1]] = a

        return block_matrix
    
import scipy.stats
def print_statistics(x):
    print("Mean:", "{:4f}".format(np.mean(x)), "Sigma:", "{:4f}".format(np.std(x)), "Skew:", "{:4f}".format(scipy.stats.skew(x, axis=None)), "Kurtosis:", "{:4f}".format(scipy.stats.kurtosis(x, axis=None)))
    print("Min:", "{:4f}".format(np.min(x)), "Max:", "{:4f}".format(np.max(x)))
    percentiles = [ 99, 95, 90, 80, 70 ]
    ps_vals = np.percentile(np.abs(x), percentiles)
    for i, ps in enumerate(percentiles):
        
        print(ps, "percentile:", "{:4f}".format(ps_vals[i]))
    
if __name__ == "__main__":
    print_statistics(np.random.normal(size=1000, scale=1))
           
