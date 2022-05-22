import sys, yaml, os

FILE_ROOT = sys.argv[1]

with open(sys.argv[2]) as f:
    global_yaml = yaml.load(f, Loader=yaml.FullLoader)
NITER = global_yaml["niter"]
    


with open(sys.argv[3]) as f:
    sampler_yaml = yaml.load(f, Loader=yaml.FullLoader)
    
sampler_yaml["file_root"] = FILE_ROOT
sampler_yaml["niter"] = NITER
sampler_yaml["orig_yaml"] = os.path.basename(sys.argv[3])

# Dump new sims parameters
stream = open(sys.argv[4], "w")
yaml.dump(sampler_yaml, stream, default_flow_style=False)  
stream.close()
   
 
