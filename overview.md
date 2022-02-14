# Overview

The sampler operates on radio telescope interferometric data. The data must be calibrated so that as well as the observed visibilities d there are model visibilities V, for every baseline, and gains g for every antenna. Calibration implies that the following equation has been fitted:

d<sub>ij</sub> = g<sub>i</sub>g<sub>j</sub>* V<sub>ij</sub>

for every baseline ij and antennas i and j.

The sampler takes as input simulated or real data, and produces the a probability distribution for g and V that will fit the data d. The maximum likelihood values for g and V may give improved calibration of the data. These values can be examined using various plotting functions.

## Typical workflow

The top-level interface is the Sampler class in sampler.py. The steps for a typical sampling run are:

1. Create a Sampler object, specifying options that modify the behaviour.
2. Load a simulation or data into the object.
3. Set priors for g and V, these will constrain their allowable range.
4. Run the sampling.
5. Examine the results.

## Data input

Currently only simulated data can be used. Simple simulated data can be created using the VisSim object in vis_creator.py. It generates random values for d, g, and V so that the above equation is satisfied.

It is also possible to load HERA simulated data. Specifically, simulations made with the non-redundant-pipeline. The data exists in files produced by the pipeline, and these have to be loaded using the VisCal and VisTrue classes in vis_creator.py. 

The user does not call VisSim, VisCal, VisTrue directly, but calls function in Sampler to load the simulations. 

## Major functionality

[Sampler class](sampler.md)

[Simulator objects](sim_classes.md)


