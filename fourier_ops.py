import numpy as np
from calcs import split_re_im, unsplit_re_im, remove_x_im, restore_x_im

SQRT_2 = np.sqrt(2)

class FourierOps:
    
    def __init__(self, ntime, nfreq, nant):
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        
    def fft2_normed(self, data):
        return np.fft.fft2(data)/data.shape[0]
    
    def ifft2_normed(self, data):
        return np.fft.ifft2(data)*data.shape[0]

    def rfft2_normed(self, data):
        return SQRT_2*np.fft.fft2(data)/data.shape[0]
    
    def rifft2_normed(self, data):
        return (np.fft.ifft2(data)*data.shape[0])/SQRT_2
 
    def condense_real_fft(self, a):
        # a: complex 2-D FFT of real values

        assert a.shape[0] == a.shape[1]

        N = a.shape[0]
        assert N%2 == 0

        # Extract the uniq complex values, picking them out of a
        unique = a[0, :N//2+1]     # From first row
        for i in range(1, N//2):                                              # from subsequent rows up to Nyquist
            unique = np.append(unique, a[i])

        unique = np.append(unique, a[N//2, :N//2+1])                          # First few in Nyqust row

        # Split into real/imag and get rid of 4 imaginary zeros
        where = [ 1, N+1, N**2-N+3, N**2+3 ]
        unique = np.delete(split_re_im(unique), where)

        assert unique.size == N**2

        return unique
    
    def F_v(self, v):
        # v is a vector of x values split into re/im and flattened by time/freq
        
        result = np.reshape(v, (self.ntime, self.nfreq, self.nant*2-1))    # Reshape so can add missing x
        result = restore_x_im(result)              # Add the missing x value for each time/freq. (Is missing for DOF fix). It is 0.
        result = unsplit_re_im(result)             # 2 lines reform to complex N-D arrayfor a Fourier transform 
        result = np.moveaxis(result, 2, 0)         # Make the axes (self.nant, self.ntime, self.nfreq) i.e. ant first. The array for the last ant has only real values.
        result = self.fft2_normed(result)                  
        condensed = self.condense_real_fft(result[-1])       # The last antenna FFT which is of real values needs to be stripped to unique values
        result = np.append(split_re_im(np.ravel(result[:-1])), condensed)                # flatten and split, now length is (self.nant-1)*self.ntime*self.nfreq*2+1*self.ntime*self.nfreq
        assert result.size == (self.nant-1)*self.ntime*self.nfreq*2+self.ntime*self.nfreq

        return result                           # This is a vector of FFT(x) with no values missing

    def F_M(self, M):
        # The columns of M are x-like, x values split into re/im and flattened by time/freq
        
        result = np.empty(M.shape)
        for j in range(M.shape[1]):
            result[:, j] = self.F_v(M[:, j])
        
        return result        # Each column of the result is now the FFT of each column of M                   
    
    def F_inv_fft(self, fft):
        
        expanded = self.expand_real_fft(fft[-self.ntime*self.nfreq:])     # Generate the full FFT of the last antenna

        result = unsplit_re_im(fft[:-self.ntime*self.nfreq])                       # 3 lines reshape to complex array (self.nant-1, self.ntime, self.nfreq)
        result = np.reshape(result, (self.nant-1, self.ntime, self.nfreq))    
        result = np.append(result, [expanded], axis=0)                     # Tack on last antenna
        result = self.ifft2_normed(result)    
        assert np.sum(result[3].imag) == 0           # All the imag should be 0
        result = np.moveaxis(result, 0, 2)           # Make the axes (self.ntime, self.nfreq, self.nant)
        result = remove_x_im(split_re_im(result))    # Remove one imaginary x value in each time/freq   
        result = np.ravel(result)                    # Flatten, now length is self.ntime*self.nfreq*(self.nant*2-1)
        return result

    def F_inv_M_fft(self, fft):
        
        # The columns of M are x-like, x values split into re/im and flattened by time/freq
        
        result = np.empty(fft.shape)
        for j in range(fft.shape[1]):
            result[:, j] = self.F_inv_fft(fft[:, j])
        
        return result        # Each column of the result is now the FFT of each column of M                   

    def AFT_Ninv_v(self, A, N, v, sqrt_N=False):          
        """ 
        AF.T Ninv multiplied by some vector. N is a vector. The vector v has to be d-like and returns FFT of x-like as vector. 

        AF is an inverse FFT followed by the old A. AF.T is the old A.T followed by an FFT, (in procedural terms).
        It's too tricky at this stage to generate AF.T N_inv itself, which could be multiplied into a 
        vector later.
        """
                                      

        if sqrt_N:
            result = print(A*(v/N))                # when using the diagonal of an array, must use * in the right way
        else:
            result = print(A*(v/np.sqrt(N))) 
        result = np.reshape(result, (self.ntime, self.nfreq, self.nant*2-1))    # Reshape so can add missing x
        result = restore_x_im(result)              # Add the missing x value for each time/freq. (Is missing for DOF fix). It is 0.
        result = unsplit_re_im(result)             # 2 lines reform to complex N-D arrayfor a Fourier transform 
        result = np.moveaxis(result, 2, 0)         # Make the axes (self.nant, self.ntime, self.nfreq) i.e. ant first. The array for the last ant has only real values.
        result = self.fft2_normed(result)                  
        condensed = self.condense_real_fft(result[-1])       # The last antenna FFT which is of real values needs to be stripped to unique values
        result = np.append(split_re_im(np.ravel(result[:-1])), condensed)                # flatten and split, now length is (self.nant-1)*self.ntime*self.nfreq*2+1*self.ntime*self.nfreq
        assert result.size == (self.nant-1)*self.ntime*self.nfreq*2+self.ntime*self.nfreq

        return result                           # This is a vector of FFT(x) with no values missing

    # Why we can't just put zeros in S for the real FFT components. Because the FFT
    # components are more than just zeros, they have to be conjugates of each other and
    # ordered. This can't be done with the prior.

    def expand_real_fft(self, components):
        N = int(np.sqrt(components.size))
        assert N**2 == components.size, "components is not square: "+str(components.shape)

        # Insert 4 zero values 
        where = [ 1, N, components.size-N+1, components.size]
        unique = np.insert(components, where, [0, 0, 0, 0])

        # Make complex
        unique = unsplit_re_im(unique)

        # Assemble
        fft = np.zeros((N, N), dtype=np.complex128)

        # First row
        fft[0, :N//2+1] = unique[:N//2+1]
        fft[0, N//2+1:N] = np.conj(np.flip(fft[0, 1:N//2]))

        # Subsequent rows half way down and their pair
        start = N//2+1
        for i in range(1, N//2):
            row_start = start+N*(i-1)
            fft[i] = unique[row_start: row_start+N]
        fft[N//2+1:N, 1:N] = np.conj(np.flip(fft[1:N//2, 1:N]))


        # Nyquist row
        fft[N//2, :N//2+1] = unique[-N//2-1:]
        fft[N//2, N//2+1:N] = np.conj(np.flip(fft[N//2, 1:N//2]))

        # The last elements in the first column beyond N//2
        fft[-N//2+1:, 0] = np.conj(np.flip(fft[1:N//2, 0]))

        return fft        


    def AF_v(self, A, fft):          
        """ 
        AF multiplied by some vector. The vector has to be FFT of x-like (x gain offsets)
        flattened. They MUST be generated by AFT_Ninv_v or have same format.

        AF is an inverse FFT followed by the old A (in procedural terms).
        """

        expanded = self.expand_real_fft(fft[-self.ntime*self.nfreq:])     # Generate the full FFT of the last antenna

        result = unsplit_re_im(fft[:-self.ntime*self.nfreq])                       # 3 lines reshape to complex array (self.nant-1, self.ntime, self.nfreq)
        result = np.reshape(result, (self.nant-1, self.ntime, self.nfreq))    
        result = np.append(result, [expanded], axis=0)                     # Tack on last antenna
        result = self.ifft2_normed(result)    
        assert np.sum(result[3].imag) == 0           # All the imag should be 0
        result = np.moveaxis(result, 0, 2)           # Make the axes (self.ntime, self.nfreq, self.nant)
        result = remove_x_im(split_re_im(result))    # Remove one imaginary x value in each time/freq   
        result = np.ravel(result)                    # Flatten, now length is self.ntime*self.nfreq*(self.nant*2-1)
        result = np.dot(A, result)                   # Now length is self.ntime*self.nfreq*nvis*2, if A is the projection matrix

        return result

