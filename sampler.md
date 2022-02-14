## Sampler class

Handles everything and is the user interface. Created as Sampler(...) with parameters as specified in the code documentation.

### Attributes

The user can access these but must not change them. There are other attributes that normally wouldn't be accessed.

vis_redcal 
> The primary location of input data.  A simulator object.
> If a simple simulation has been loaded (load\_sim()), then all the values are invented. vis_redcal will be of type VisSim.

> If a HERA non-redundant-pipeline simulation has been loaded (load\_nr\_sim()), then this contains the RedCal calibration gains g and model V, as well as the observed visibililties. It also contains the noise on the data, which is used in the sampling (N matrix). vis\_redcal will be of type VisCal which is derived from VisSim.
 

vis_true
> A simulator object. If a simple simulation has been loaded, this will be the same as vis\_redcal.
>.
> If a HERA non-redundant-pipeline simulation has been loaded (load\_nr\_sim()), then this contains the true model and gains. What these mean will be described in the class descriptions.

samples
> This is dictionary and contains the samples taken during the Gibbs sampling. The samples are in order of the sampling iterations, but burn-in has been removed. The dictionary contains "x", "g" and "V" samples. 
A "sample" is a 3-D array of shape (ntime, nfreq, nant or nbaseline). There is a sample for each iteration of the Gibbs sampling. The values are complex. Example: samples["V"][1000] gives an array which is the V values from the 1000'th iteration of the Gibbs sampling (after burn-in). The V values array is of shape (nant, nfreq, nbaseline).
> 
> The gains g are not actually sampled, it is the gain perturbations x that are sampled. However, new gains can be calculated from the new sampled x values using g = g_bar*(1+x).
> 
> The sampling has to be run to get the samples.

vis_sampled
>This is a copy of vis\_redcal, but there are two changes: the x values are the "best" obtained from the sampling, the same for the V values. So x and V are changed to contain the results from sampling, all other simulation values are unaltered. What "best" means depends on how this was specified when Sampler was created.



## Methods

These fall into 2 categories: setting up and running the sampler, and looking at the results. Most ways of looking at the results are via plotting methods. These methods can be run by a user:

```
    def load_nr_sim(self, file_root, time_range=None, freq_range=None, remove_redundancy=False, initial_solve_for_x=False):
    def load_sim(self, nant, ntime=1, nfreq=1, initial_solve_for_x=False, **kwargs):
    def set_S_and_V_prior(self, S, V_mean, Cv):
    def nant(self):
    def nvis(self):
    def run(self):
    def plot_marginals(self, parameter, cols, time=None, freq=None, which=[ "True", "Redcal", "Sampled" ]):
        def plot_hist(a, fname, label, sigma_prior, other_vals, index):
    def plot_corner(self, parameters, time=None, freq=None, threshold=0.0, xgs=None, Vs=None):
    def plot_trace(self, parameter, time=None, freq=None, index=None):
    def print_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0, list_them=False, count_them=False):
    def plot_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0):
    def plot_sample_means(self, parameters, time=None, freq=None):
    def plot_results(self, time=None, freq=None): 
    def plot_gains(self, time=None, freq=None, sigma=3):
```



