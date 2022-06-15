import sys, yaml, os

FILE_ROOT = sys.argv[1]

with open(sys.argv[2]) as f:
    global_yaml = yaml.load(f, Loader=yaml.FullLoader)
NITER = global_yaml["niter"]
FIX_DEGENERACIES = global_yaml["fix_degeneracies"]
REMOVE_REDUNDANCY = global_yaml["remove_redundancy"]    


with open(sys.argv[3]) as f:
    sampler_yaml = yaml.load(f, Loader=yaml.FullLoader)
    
sampler_yaml["file_root"] = FILE_ROOT
sampler_yaml["niter"] = NITER
sampler_yaml["fix_degeneracies"] = FIX_DEGENERACIES
sampler_yaml["remove_redundancy"] = REMOVE_REDUNDANCY
sampler_yaml["orig_yaml"] = os.path.basename(sys.argv[3])

# Dump new sims parameters
stream = open(sys.argv[4], "w")
yaml.dump(sampler_yaml, stream, default_flow_style=False)  
stream.close()
   
 