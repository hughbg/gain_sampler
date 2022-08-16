import os
import pandas as pd
 

def tf(x):
    try:
        num = round(float(x), 3)
        return num
    except:
        return x

def write_cell(f, s):
    f.write(str(s)+",")

def extract(fname):
    print(fname)
    with open(fname) as f:
        lines = f.readlines()
        
    d = {}
    for l in lines:
        if "Expected chi2" in l and "print" not in l:
            d["Expected chi2"] = tf(l.split()[-1][:-4])

        if "CHI2" in l and ">" in l and "image" not in l and "print" not in l:
            ls = l.split()
            d["Calculated chi2"] = tf(ls[1])
            d["Calculated after sampling chi2"] = tf(ls[3])
            
        if "abs Log posterior" in l and "print" not in l:
            ls = l.split()
            d["Posterior redcal"] = tf(ls[3])
            d["Posterior sampled"] = tf(ls[5])

        if "RMS" in l and ">" in l and "print" not in l and "image" not in l:
            ls = l.split()
            d["RMS redcal"] = tf(ls[1])
            d["RMS sampled"] = tf(ls[3])
  
        if "Slope" in l:
            ls = l.split()
            s = (ls[0]+" "+ls[1]+" "+ls[2]).replace("\"", "").replace(":", "")
            d[s] = str(tf(ls[3]))+" +/- "+str(tf(ls[5][:-4]))

            
    return d



rows = {}

for nb in os.listdir("redundancy"):
    if nb[-5:] == "ipynb":
        if nb not in rows: rows[nb] = {}         
        rows[nb]["redundancy"] = extract("redundancy/"+nb)

for nb in os.listdir("no_redundancy"):
    if nb[-5:] == "ipynb":
        if nb not in rows: rows[nb] = {}         
        rows[nb]["no_redundancy"] = extract("no_redundancy/"+nb)
    
sub_keys = set(rows["paper_plots_vanilla_out.ipynb"]["redundancy"].keys()).union(set(rows["paper_plots_vanilla_out.ipynb"]["no_redundancy"].keys()))

# Ensure they are all there
for case in rows:
    if "redundancy" not in rows[case]: rows[case]["redundancy"] = {}
    if "no_redundancy" not in rows[case]: rows[case]["no_redundancy"] = {}
    for key in sub_keys:
        if key not in rows[case]["redundancy"]: rows[case]["redundancy"][key] = ""
        if key not in rows[case]["no_redundancy"]: rows[case]["no_redundancy"][key] = ""

# Order
sub_keys = ["Expected chi2", "Calculated chi2", "Calculated after sampling chi2", "Posterior redcal", "Posterior sampled", "RMS redcal", "RMS sampled", "Redcal Amp Slope", "Redcal Phase Slope", "Sampled Amp Slope", "Sampled Phase Slope"]

outf = open("summary.csv", "w")
outf.write("Case,Redundancy"+","*len(sub_keys)+"No redundancy"+","*len(sub_keys)+"\n")
outf.write(","+",".join(sub_keys)+","+",".join(sub_keys)+",\n")

for key in [ "paper_plots_unistretch0.01_out.ipynb"]: #rows:
    write_cell(outf, key[12:-6])
    for key_r in sub_keys:
        write_cell(outf, rows[key]["redundancy"][key_r]) 
    for key_r in sub_keys:
        write_cell(outf, rows[key]["no_redundancy"][key_r]) 
        
        
    outf.write("\n")
    
outf.close()
    
with open("summary.csv") as f:
    lines = f.readlines()

csv = [ None for i in range(len(lines)) ]

for i, l in enumerate(lines):
    csv[i] = l.split(",")[:-1]

for i, l in enumerate(csv):
    assert len(csv[i]) == len(csv[0])
    
    
# Transpose
with open("summary.csv", "w") as f:

    cols = len(csv[0])
    rows = len(csv)
    for i in range(cols):
        s = ""
        for j in range(rows):
            s += csv[j][i]
            if j < rows-1: s += ","
        f.write(s+"\n")

# to read csv file named "samplee"
a = pd.read_csv("summary.csv")
 
# to save as html file
# named as "Table"
a.to_html("summary.html")
 

    
    

    
            
            
