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
    new_a = np.empty(a.size*2)
    for i in range(a.size):
        new_a[i*2] = a[i].real
        new_a[i*2+1] = a[i].imag
    return new_a

def unsplit_re_im(a):
    new_a = np.empty(a.size//2, dtype=np.complex64)
    for i in range(new_a.size):
        new_a[i] = complex(a[i*2], a[i*2+1])
    return new_a

if __name__ == "__main__":
    print(unsplit_re_im(np.array([1, 2, 3, 4, 5, 6])))
