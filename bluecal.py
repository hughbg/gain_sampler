import matplotlib.pyplot as plt
import numpy as np
import sys
from calcs import split_re_im, unsplit_re_im
from vis_creator import VisCal, VisTrue
from gls import gls_solve


def bluecal_run(file_root):
    def rms(x, y): return np.abs(np.sqrt(np.mean((x-y)**2)))

    def plot_cals(true, calibrated, title, label1, label2, step):
        s = 3

        plt.clf()
        plt.scatter(np.real(true), np.imag(true), marker="x", label=label1, s=20)
        plt.scatter(np.real(calibrated), np.imag(calibrated), label=label2, s=s)
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.legend()
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(label1.replace(" ", "_")+str(step)+".png")
        
        plt.clf()
        plt.scatter(np.real(true-calibrated), np.imag(true-calibrated), s=s)
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title("Difference between\n"+label1+" and "+label2+"\nDiff="+str(rms(true, calibrated)), fontsize=10)
        plt.tight_layout()
        plt.savefig(label1.replace(" ","_")+"_diff_"+str(step)+".png")

        return label1.replace(" ", "_")+str(step)+".png", label1.replace(" ","_")+"_diff_"+str(step)+".png"

    def four_plots_in_table(im1, im2, im3, im4):
        return "<table><tr><td><img src="+im1+"></td>" + \
            "<td><img src="+im2+"></td>" + \
            "<td><img src="+im3+"></td>" + \
            "<td><img src="+im4+"></td></tr><table>"

    def all_plots(true, cal, step):
        obs, obs_diff = plot_cals(cal.get_calibrated_visibilities(), cal.V_model, "Calibrated observed visibilities vs.\nModified model", "Calibrated observed visibilities", "Modified model", step)       
        model, model_diff = plot_cals(true.V_model, cal.V_model, "True model vs. modified model", "True model", "Modified model", step)       
        gain, gain_diff = plot_cals(true.get_antenna_gains(), cal.get_antenna_gains(), "True gains vs. modified gains", "True gains", "Modified gains", step)
        x, x_diff = plot_cals(true.x, cal.x, "x values", "0", "Modified x", step)

        return four_plots_in_table(model, gain, obs, x)+"\n<p>"+four_plots_in_table(model_diff, gain_diff, obs_diff, x_diff)

    plt.figure(figsize=(4, 4))
    open("plot_cals.html", "w").close()

    vis_values = VisCal(file_root, time=0, freq=0)
    true_vis_values = VisTrue(file_root, time=0, freq=0)
    
    print("Chi squared calibrated:", round(vis_values.get_chi2(), 2), "Chi squared true:", round(true_vis_values.get_chi2(), 2))
    
    f = open("plot_cals.html", "a")
    f.write("<h2>After redundant calibration</h2>Get the model and gains from redundant calibration. The true gains are all the same for 1 frequency.\n")
    f.write(all_plots(true_vis_values, vis_values, -1))
    f.close()

    
    for i in range(1):

        print(i, "-----------")
        
        # Update model
        vis_values.V_model = vis_values.get_calibrated_visibilities()

        gls_solve(vis_values)

        print("Chi squared calibrated:", round(vis_values.get_chi2(), 2), "Chi squared true:", round(true_vis_values.get_chi2(), 2))

        f = open("plot_cals.html", "a")
        f.write("<h2>Step "+str(i)+"</h2>Calculate a new model then perform GLS. Note the ranges may not be the same as the previous plots.")
        f.write(all_plots(true_vis_values, vis_values, i))
        f.close()

if __name__ == "__main__":
    
    bluecal_run(sys.argv[1])
