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
> If a HERA non-redundant-pipeline simulation has been loaded (load\_nr\_sim()), then this contains the true model and gains. What these mean will be described in the [class descriptions](sim_classes.md).

samples
> This is dictionary and contains the samples taken during the Gibbs sampling. The samples are in order of the sampling iterations, but burn-in has been removed. The dictionary contains "x", "g" and "V" samples. 
A "sample" is a 3-D array of shape (ntime, nfreq, nant or nbaseline). There is a sample for each iteration of the Gibbs sampling. The values are complex. Example: samples["V"][1000] gives an array which is the V values from the 1000'th iteration of the Gibbs sampling (after burn-in). The V values array is of shape (nant, nfreq, nbaseline).
> 
> The gains g are not actually sampled, it is the gain perturbations x that are sampled. However, "gain samples" are calculated from the new sampled x values using g = g_bar*(1+x).
> 
> The sampling has to be run to get the samples.

vis_sampled
>This is a copy of vis\_redcal, but there are two changes: the x values are the "best" obtained from the sampling, and the same for the V values. So x and V are changed to contain the results from sampling, all other simulation values are unaltered. What "best" means depends on how this was specified when Sampler was created.


To summarize  the 3 visibility data objects:




If a simple sim is loaded via load\_sim(), the 3 visibility objects contain:

|vis\_redcal   and vis\_true | vis\_sampled| Comment |
| :----------- | :------------------: |:---: |
| V\_obs\_sim     | V\_obs\_sim | The same |
| g\_bar\_sim   | g\_bar\_sim| The same |
| x\_sim          | x\_samp  | Not the same |
| V\_sim     | V\_samp | Not the same |


If a non-redundant-pipeline sim is loaded via load\_nr\_sim():

|vis\_redcal      |vis\_true | vis\_sampled| Comment |
| :----------- | :------------------: | :------------:| :---: |
| V\_obs\_sim      | V\_obs\_sim       | V\_obs\_sim | All the same |
| g\_bar\_cal   | g\_bar\_sim     | g\_bar\_cal| Sampled has the same as vis\_redcal |
| x = 0       |   x = 0    | x\_samp  |  x exists only after sampling
| V\_cal   | V\_sim  | V\_samp | Not the same |


All these values can be compared against each other.



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
    def plot_corner(self, parameters, time=None, freq=None, threshold=0.0, xgs=None, Vs=None):
    def plot_trace(self, parameter, time=None, freq=None, index=None):
    def print_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0, list_them=False, count_them=False):
    def plot_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0):
    def plot_sample_means(self, parameters, time=None, freq=None):
    def plot_results(self, time=None, freq=None): 
    def plot_gains(self, time=None, freq=None, sigma=3):
```



