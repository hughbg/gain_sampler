import os, re
import numpy as np


cases = {

    "paper_plots_side0.2_out.ipynb": "1(a), 0.2",
    "paper_plots_unistretch0.01_out.ipynb": "3(b)",    # But why 0.01 should be 0.02
    "paper_plots_4a_0.02_out.ipynb": "4(a), 0.02",
    "paper_plots_4c_a_out.ipynb" : "4(c)(i)", 
    "paper_plots_4b_0.02_out.ipynb" : "4(b), 0.02", 
    "paper_plots_4c_b_out.ipynb" : "4(c)(ii)", 
    "paper_plots_4a_0.01_out.ipynb": "4(a), 0.01", 
    "paper_plots_stretch0.01_out.ipynb" : "3(a) 0.01", 
    "paper_plots_vanilla_out.ipynb" : "0", 
    "paper_plots_stretch0.02_out.ipynb": "3(a) 0.01", 
    "paper_plots_outlier7_1.1_out.ipynb": "3(d)", 
    "paper_plots_outlier2_1.1_out.ipynb" : "3(c)", 
    "paper_plots_4b_0.01_out.ipynb" : "4(b), 0.01", 
    "paper_plots_side0.05_out.ipynb" : "1(a), 0.05", 
    "paper_plots_stretch0.02.ipynb": "3(a) 0.02",

    "paper_plots_high_diffuse_out.ipynb": "source case 1",
    "paper_plots_one_bright_out.ipynb": "source case 2",
    "paper_plots_vanilla_huge_prior_out.ipynb" : "prior case 1",
    "paper_plots_vanilla_flat_prior_out.ipynb" : "prior case 2",
    "paper_plots_vanilla_all_modes_gauss_out.ipynb" : "prior case 3",
    "paper_plots_vanilla_all_modes_gauss_out.ipynb" : "prior case 4",
    "paper_plots_vanilla_wide_prior_out.ipynb" : "prior case 5",
    "paper_plots_vanilla_4_modes_gauss_out.ipynb" : "prior case 6",
    "paper_plots_vanilla_Vprior_offset_out.ipynb" : "prior case 7"
}

codes = {
    "A" : "Improved amp and phase cal",
    "HA": "Impoved amp cal, phase same",
    "HP": "Impoved phase cal, amp same",
    "MA": "Mixed, amp improved, phase worse",
    "MP": "Mixed, phase improved, amp worse",
    "S" : "Same"
}

latex_codes1 = {
    "A" : "\\makecell{{\\small Improved amp}\\\\{\\small and phase cal}}",
    "HA": "{\\small Improved amp cal}",
    "HP": "{\\small Improved phase cal}",
    "MA": "\\makecell{{\\small Mixed, amp improved}\\\\{\\small phase worse}}",
    "MP": "\\makecell{{\\small Mixed, phase improved}\\\\{\\small amp worse}}",
    "S" : "Same"
}

latex_codes = {    # Fo r Gibbs vs redcal
    "A" : "$\\downarrow\\downarrow$",
    "HA": "$\\downarrow =$",
    "HP": "$=\\downarrow$",
    "MA": "$\\downarrow\\uparrow$",
    "MP": "$\\uparrow\\downarrow$",
    "S" : "=="
}





def check(fname, tags):
    print(fname)
    precision = pre = 4
    with open(fname) as f:
        lines = f.readlines()
        
    for tag in tags:
        rms_redcal_amp = rms_redcal_phase = rms_sampled_amp = rms_sampled_phase = -1
        for l in lines:
            
            if tag+" rms true diff redcal amp" in l and "print" not in l and "image" not in l:
                ls = l.split()
                rms_redcal_amp = np.round(float(re.sub("[^0-9.]", "", ls[-1])), pre)

            if tag+" rms true diff redcal phase" in l and "print" not in l and "image" not in l:
                ls = l.split()
                rms_redcal_phase = np.round(float(re.sub("[^0-9.]", "", ls[-1])), pre)

            if tag+" rms true diff sampled amp" in l and "print" not in l and "image" not in l:
                ls = l.split()
                rms_sampled_amp = np.round(float(re.sub("[^0-9.]", "", ls[-1])), pre)

            if tag+" rms true diff sampled phase" in l and "print" not in l and "image" not in l:
                ls = l.split()
                rms_sampled_phase = np.round(float(re.sub("[^0-9.]", "", ls[-1])), pre)
                

        print("\t"+tag, end=" ")
        if -1 in [ rms_redcal_amp, rms_sampled_amp, rms_redcal_phase, rms_sampled_phase ]:
            print("Unknown")
            code = "U"
        else:
            if rms_redcal_amp > rms_sampled_amp and rms_redcal_phase > rms_sampled_phase:
                #print("All better amp:", "redcal amp =", rms_redcal_amp, "sampled amp =", rms_sampled_amp, "redcal phase =" ,rms_redcal_phase, "sampled phase =", 
                #      rms_sampled_phase, "(", rms_redcal_amp-rms_sampled_amp, ",", rms_redcal_phase-rms_sampled_phase, ")", end=" ")
                code = "A"
            elif rms_redcal_amp < rms_sampled_amp and rms_redcal_phase < rms_sampled_phase: 
                #print("Worse", "redcal amp =", rms_redcal_amp, "sampled amp =", rms_sampled_amp, "redcal phase =", rms_redcal_phase, "sampled phase =", rms_sampled_phase, end= " ")
                code = "W"
            elif rms_redcal_amp == rms_sampled_amp and rms_redcal_phase == rms_sampled_phase:
                #print("Same", "redcal amp =", rms_redcal_amp, "sampled amp =", rms_sampled_amp, "redcal phase =", rms_redcal_phase, "sampled phase =", rms_sampled_phase, end= " ")
                code = "S"
            elif ( rms_redcal_amp > rms_sampled_amp and rms_redcal_phase >= rms_sampled_phase ) or \
                    ( rms_redcal_phase > rms_sampled_phase and rms_redcal_amp >= rms_sampled_amp ):
                #print("Half better", "redcal amp =", rms_redcal_amp, "sampled amp =", rms_sampled_amp, "redcal phase =", rms_redcal_phase, "sampled phase =", rms_sampled_phase, "(", rms_redcal_amp-rms_sampled_amp, ",", rms_redcal_phase-rms_sampled_phase, ")", end= " ")
                if rms_redcal_amp > rms_sampled_amp and rms_redcal_phase >= rms_sampled_phase: code = "HA"
                else: code = "HP"
            else: 
                #print("Mixed", "redcal amp =", rms_redcal_amp, "sampled amp =", rms_sampled_amp, "redcal phase =", rms_redcal_phase, "sampled phase =", rms_sampled_phase, end= " ")
                if rms_redcal_amp > rms_sampled_amp: code = "MA"
                else: code = "MP"
            print()
            
    return {
        "tag": tag,
        "code": code,
        "rms_redcal_amp": rms_redcal_amp,
        "rms_redcal_phase": rms_redcal_phase,
        "rms_sampled_amp": rms_sampled_amp,
        "rms_sampled_phase": rms_sampled_phase
    } 

summary = { "redundant" : {}, "non-redundant": {} }
for nb in os.listdir("redundancy"):
    if nb[-5:] == "ipynb":
        assert nb in cases, nb
        res = check("redundancy/"+nb, ["CAL"])
        summary["redundant"][cases[nb]] = res

for nb in os.listdir("no_redundancy"):
    if nb[-5:] == "ipynb":
        assert nb in cases, nb
        res = check("no_redundancy/"+nb, ["CAL"])
        summary["non-redundant"][cases[nb]] = res
        
# Ensure the same
for key in summary["redundant"]:
    if key not in summary["non-redundant"]: 
        summary["non-redundant"][key] = "U"
        
for key in summary["non-redundant"]:
    if key not in summary["redundant"]: 
        summary["redundant"][key] = "U"
        

      
"""
# print
    
for key in summary["redundant"]:
    if summary["redundant"][key] != summary["non-redundant"][key]:
        nr_diff = "nr diff"
    else:
        nr_diff = "nr no diff"
    print(key, "|", summary["redundant"][key], "|", summary["non-redundant"][key], nr_diff)
"""

# latex

print()
best = ""
best_val = -1e39
for key in sorted(summary["redundant"]):
    if "prior" not in key:
        print(key, "&", end=" ")
        for field in [ "rms_redcal_amp", "rms_redcal_phase", "rms_sampled_amp", "rms_sampled_phase" ]:
            print(summary["redundant"][key][field], "&", end=" ")
        print(latex_codes[summary["redundant"][key]["code"]], "&", end=" ")
        for field in [ "rms_sampled_amp", "rms_sampled_phase" ]:
            print(summary["non-redundant"][key][field], "&", end=" ")
        print(latex_codes[summary["non-redundant"][key]["code"]], "&", end=" ")
        
        # Last column
        if summary["redundant"][key]["rms_sampled_amp"] == summary["non-redundant"][key]["rms_sampled_amp"]:
            first = "="
        elif summary["redundant"][key]["rms_sampled_amp"] > summary["non-redundant"][key]["rms_sampled_amp"]:
            first = "$\\downarrow$"
        else: first = "$\\uparrow$"
            
        if summary["redundant"][key]["rms_sampled_phase"] == summary["non-redundant"][key]["rms_sampled_phase"]:
            second = "="
        elif summary["redundant"][key]["rms_sampled_phase"] > summary["non-redundant"][key]["rms_sampled_phase"]:
            second = "$\\downarrow$"
        else: second = "$\\uparrow$"
        print(first+second)
            
            
        print("\\\\")
        print("\\hline")
        
        # Best
        for r in [ "redundant", "non-redundant" ]:
            for component in [ "amp", "phase" ]:
                if summary[r][key]["rms_redcal_"+component]-summary[r][key]["rms_sampled_"+component] > best_val:
                    best_val = summary[r][key]["rms_redcal_"+component]-summary[r][key]["rms_sampled_"+component]
                    best = key+" "+r+" "+component+" "+str(summary[r][key]["rms_redcal_"+component])+" "+str(summary[r][key]["rms_sampled_"+component])
                    
print("Best:", best)
        
print()
best = ""
best_val = -1e39
for key in sorted(summary["redundant"]):
    if "prior" in key:
        print(key, "&", end=" ")
        for field in [ "rms_redcal_amp", "rms_redcal_phase", "rms_sampled_amp", "rms_sampled_phase" ]:
            print(summary["redundant"][key][field], "&", end=" ")
        print(latex_codes[summary["redundant"][key]["code"]], "&", end=" ")
        for field in [ "rms_sampled_amp", "rms_sampled_phase" ]:
            print(summary["non-redundant"][key][field], "&", end=" ")
        print(latex_codes[summary["non-redundant"][key]["code"]], "&", end=" ")
        if summary["redundant"][key]["code"] == summary["non-redundant"][key]["code"]: print("No", end=" ")
        else: print("Different", end=" ")    
        print("\\\\")
        print("\\hline")
        
        # Best
        for r in [ "redundant", "non-redundant" ]:
            for component in [ "amp", "phase" ]:
                if summary[r][key]["rms_redcal_"+component]-summary[r][key]["rms_sampled_"+component] > best_val:
                    best_val = summary[r][key]["rms_redcal_"+component]-summary[r][key]["rms_sampled_"+component]
                    best = key+" "+r+" "+component+" "+str(summary[r][key]["rms_redcal_"+component])+" "+str(summary[r][key]["rms_sampled_"+component])
                    
print("Best:", best)


    