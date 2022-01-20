import numpy as np

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

class BlockMatrix:
    
    def __init__(self):
        self.matrices_incoming = []
    
    def add(self, a, replicate=1):
        assert len(a.shape) <= 2, "Can't assemble block matrix with dimensions > 2"
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
    
class Alonso:
    def __init__(self, num, alpha=0, xi=1):
        nu_ref = 400
        
        alonso = lambda nu1, nu2: (nu_ref**2/nu1/nu2)**alpha*np.exp(-np.log(nu1/nu2)**2/(2*xi**2))
            
        # Setup two-step indexing
        a = np.array_split(np.arange(405, 945, 1), num)
        nu = [ np.mean(sub_a) for sub_a in a ]
        
        self.cov = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                self.cov[i, j] = alonso(nu[i], nu[j])
                
        assert np.allclose(self.cov, self.cov.T)    # Check is symmetric
        
        val, vec = np.linalg.eig(self.cov)
        if np.any(val<0):
            val = np.where(val<0, 0, val)
        
            self.cov = np.dot(vec, np.dot(np.diag(val), np.linalg.inv(vec)))

                
    def average_signal(self, niter=10000):
        samples = np.zeros((niter, self.cov.shape[0]))
        mean = np.zeros(self.cov.shape[0])
        for i in range(niter):
            samples[i] = np.random.multivariate_normal(mean, C)

        return np.mean(samples, axis=0)
    
def get_cov_matrix(ntime, nfreq, nant, diag=1.0, alpha=0, xi=1, correlate_time=False, 
                correlate_freq=False, occupancy=False):
    order = ("ntime", "nfreq", "nant")  
    
    def flat(t, f, a):
        # Always time, freq, ant input, regardless of above order
        layout = [ 0, 0, 0 ]
        indexes = [ 0, 0, 0 ]
        
        where = order.index("ntime")
        layout[where] = ntime
        indexes[where] = t
        where = order.index("nfreq")
        layout[where] = nfreq
        indexes[where] = f
        where = order.index("nant")
        layout[where] = nant
        indexes[where] = a
        return np.ravel_multi_index(indexes, layout)
    
    def indexes(i):
        layout = [ 0, 0, 0 ]
        
        where = order.index("ntime")
        layout[where] = ntime
        where = order.index("nfreq")
        layout[where] = nfreq
        where = order.index("nant")
        layout[where] = nant
        
        ind = np.unravel_index(i, layout)
        
        # Always time, freq, ant, output, regardless of above order
        return ind[order.index("ntime")], ind[order.index("nfreq")], ind[order.index("nant")]
    

    # Setup matrix for nant
    if occupancy:
        cov_f = np.full((nfreq, nfreq), 1)
        if correlate_time: cov_t = np.full((ntime, ntime), 1)
    else:
        cov_f = Alonso(nfreq, alpha, xi).cov
        if correlate_time: cov_t = Alonso(ntime, alpha, xi).cov
    
    data = np.zeros((nant*ntime*nfreq, nant*ntime*nfreq))

      
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ti, fi, ai = indexes(i)
            tj, fj, aj = indexes(j)

            if correlate_time and ti != tj and fi == fj and ai == aj:       
                    data[flat(ti, fi, ai), flat(tj, fj, aj)] = cov_t[ti, tj]
            if correlate_freq and ti == tj and fi != fj and ai == aj:       
                data[flat(ti, fi, ai), flat(tj, fj, aj)] = cov_f[fi, fj]
            if i == j:
                data[i, j] = cov_f[ti, tj]
                

                            
    return data*diag/np.max(data)

            

if __name__ == "__main__":
    bm = BlockMatrix()
    bm.add(np.array([1, 1]))
    bm.add(np.array([[2,2,2],[2,2]]))
    bm.add(np.array([1, 1]))
    print(bm.assemble())
           
