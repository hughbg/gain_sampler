import numpy as np
from matplotlib import use; use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 12]


def plot_compare(_x_in, _x_out, fname, x_lab="x_in", y_lab="x_out"):

  assert _x_in.shape == _x_out.shape, "x_in, x_out should be the same shape"
  assert _x_in.shape[1]%2 == 0, "number of x values for each trial should be even"

  x_in = np.abs(_x_in)
  x_out = np.abs(_x_out)

  plt.clf()

  fig, ax = plt.subplots(x_in.shape[1]//2, 2)
  for i in range(x_in.shape[1]):
    #figure = corner.corner(bias)
    #plt.savfig("bias.png")
    plt.subplot(x_in.shape[1]//2, 2, i+1)
    plt.hist2d(x_in[:, i], x_out[:, i], range=[ [np.min(x_in), np.max(x_in)], [np.min(x_out), np.max(x_out)] ],  bins=50, cmin=1)

    plt.plot([np.min(x_in), np.max(x_in)], [np.min(x_in), np.max(x_in)], "r", lw=0.6)
    plt.xlabel(x_lab[0]+str(i)+" in amplitude")
    plt.ylabel(x_lab[0]+str(i)+" out amplitude")
    plt.title(x_lab[0]+str(i))

  fig.tight_layout()
  plt.savefig(fname)
  np.savetxt(x_lab+".dat", x_in)
  np.savetxt(y_lab+".dat", x_out)
  plt.close()

  x_in = np.angle(_x_in)
  x_out = np.angle(_x_out)

  plt.clf()
  
  fig, ax = plt.subplots(x_in.shape[1]//2, 2)
  for i in range(x_in.shape[1]):
    #figure = corner.corner(bias)
    #plt.savfig("bias.png")
    plt.subplot(x_in.shape[1]//2, 2, i+1)
    plt.hist2d(x_in[:, i], x_out[:, i], range=[ [np.min(x_in), np.max(x_in)], [np.min(x_out), np.max(x_out)] ],  bins=50, cmin=1)

    plt.plot([np.min(x_in), np.max(x_in)], [np.min(x_in), np.max(x_in)], "r", lw=0.6)
    plt.xlabel(x_lab[0]+str(i)+" in phase")
    plt.ylabel(x_lab[0]+str(i)+" out phase")
    plt.title(x_lab[0]+str(i))

  fig.tight_layout()
  plt.savefig(fname[:-4]+"_phase.png")
  np.savetxt(x_lab+"_phase.dat", x_in)
  np.savetxt(y_lab+"_phase.dat", x_out)
  plt.close()


  

