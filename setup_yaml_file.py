import sys, yaml, os

FILE_ROOT = sys.argv[1]

with open(sys.argv[2]) as f:
    global_yaml = yaml.load(f, Loader=yaml.FullLoader)

with open(sys.argv[3]) as f:
    global_sampler_yaml = yaml.load(f, Loader=yaml.FullLoader)


NITER = global_sampler_yaml["niter"]
FIX_DEGENERACIES = global_sampler_yaml["fix_degeneracies"]
REMOVE_REDUNDANCY = global_sampler_yaml["remove_redundancy"]    
SMOOTH_GAINS = global_yaml["smooth_gains"]


with open(sys.argv[4]) as f:
    sampler_yaml = yaml.load(f, Loader=yaml.FullLoader)
    
sampler_yaml["file_root"] = FILE_ROOT
sampler_yaml["niter"] = NITER
sampler_yaml["fix_degeneracies"] = FIX_DEGENERACIES
sampler_yaml["remove_redundancy"] = REMOVE_REDUNDANCY
sampler_yaml["smooth_gains"] = SMOOTH_GAINS
sampler_yaml["orig_yaml"] = os.path.basename(sys.argv[4])

# Dump new sims parameters
stream = open(sys.argv[5], "w")
yaml.dump(sampler_yaml, stream, default_flow_style=False)  
stream.close()
   
 
