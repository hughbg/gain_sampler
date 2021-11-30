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
    if len(a.shape) == 1:
        new_a = np.empty(a.size*2)
        for i in range(a.size):
            new_a[i*2] = a[i].real
            new_a[i*2+1] = a[i].imag
    else:
        # Split 2-d with the first axis the number of samples, second axis being the values
        new_a = np.zeros((a.shape[0], a.shape[1]*2))
        for i in range(a.shape[1]):
            new_a[:, 2*i] = np.real(a[:, i])
            new_a[:, 2*i+1] = np.imag(a[:, i])
        
    return new_a

def unsplit_re_im(a):
    if len(a.shape) == 1:
        new_a = np.empty(a.size//2, dtype=np.complex64)
        for i in range(new_a.size):
            new_a[i] = complex(a[i*2], a[i*2+1])
    elif len(a.shape) == 2:
        # Unsplit 2-d with the first axis the number of samples, second axis being the values
        new_a = np.zeros((a.shape[0], a.shape[1]//2), dtype=np.complex64)
        for i in range(new_a.shape[1]):
            new_a[:, i] = a[:, 2*i]+a[:, 2*i+1]*1j
    else:
        raise RuntimeError("Don't know how to unsplit shape "+str(a.shape))
    return new_a

if __name__ == "__main__":
    a = unsplit_re_im(np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]))
    print(a)
    print(split_re_im(a))
