from resource import getrusage, RUSAGE_SELF
from sampler import Sampler
from s_manager import SManager
import os, time, hickle, cProfile
import numpy as np
import sys, yaml

def all_chi2(vis):
    print(cfg["file_root"]+":", "Expected chi2", np.mean(vis.chi2["Jee"][0]))
    if vis.is_redundant: print(cfg["file_root"]+":", "calculated chi2 by redcal", vis.get_chi2(over_all=True))
    for dof in [ "default", "as_if_redundant", "as_if_non_redundant", "use_nvis" ]:
        for use_noise in [ True, False ]:
            print(cfg["file_root"]+":", "calculated chi2 by equation, dof", dof, "use_noise", use_noise, vis.get_chi2_eqn(dof=dof, use_noise=use_noise))


np.seterr(invalid='raise')

with open(sys.argv[1]) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

sampler = Sampler(seed=99, niter=cfg["niter"], burn_in=10, best_type=cfg["best_type"], best_use=cfg["best_use"],
                  best_measure=cfg["best_measure"], random_the_long_way=True, use_conj_grad=True, 
                  report_every=cfg["report_every"])
sampler.load_nr_sim(cfg["file_root"], with_redcal=cfg["with_redcal"])  

all_chi2(sampler.vis_redcal)

if cfg["fix_degeneracies"]: sampler.fix_redcal_degeneracies()
    
all_chi2(sampler.vis_redcal)
    
if cfg["remove_redundancy"]: sampler.vis_redcal.remove_redundancy()
    
# Fourier mode setup for S
if cfg["modes"]["pattern"] == "flat":
    pattern = lambda x, y: 1 
elif cfg["modes"]["pattern"] == "gauss":
    pattern = lambda x, y: np.exp(-0.5*(x**2+y**2)/.005)
elif cfg["modes"]["pattern"] == "random":
    pattern = lambda x, y: np.random.random(size=1)
else:
    raise ValueError("Invalid mode pattern")


sm = SManager(sampler.vis_redcal.ntime, sampler.vis_redcal.nfreq, sampler.vis_redcal.nant)
sm.generate_S(pattern, modes=cfg["modes"]["num"], ignore_threshold=0, zoom_from=tuple(cfg["modes"]["zoom_from"]), 
              scale=cfg["modes"]["scale"])   

# V prior
V_mean = sampler.vis_redcal.V_model+cfg["priors"]["V_offset"]
Cv_diag = np.full(V_mean.shape[2]*2, cfg["priors"]["Cv"])

sampler.set_S_and_V_prior(sm, V_mean, Cv_diag)

if cfg["smooth_gains"]: sampler.gain_smooth_modes = sm.gain_smooth_modes

start = time.time()
#cProfile.run("sampler.run()", filename="sampler.prof", sort="cumulative")

sampler.run()

print(sampler.vis_sampled.get_rms())

all_chi2(sampler.vis_sampled)
print("Run time:", time.time()-start)

case = os.path.basename(cfg["orig_yaml"][:-5]).split("_")[1:]
if len(case) > 0 :case = "_"+"_".join(case)
else: case = ""
dirname = os.path.dirname(cfg["file_root"])+"/sampled_"+os.path.basename(cfg["file_root"])+case

try:
    os.mkdir(dirname)
except: pass
np.savez_compressed(dirname+"/"+"samples", x=sampler.samples["x"], g=sampler.samples["g"], V=sampler.samples["V"])
sampler.fops = sampler.S = sampler.samples["x"] = sampler.samples["g"] = sampler.samples["V"] = None
sampler.cfg = cfg
print("redcal chi2", sampler.vis_redcal.get_chi2(over_all=True))
hickle.dump(sampler, dirname+"/sampler.hkl", mode='w', compression='gzip')

print("Wrote", dirname)
