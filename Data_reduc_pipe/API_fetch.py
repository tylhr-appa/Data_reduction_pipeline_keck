import numpy as np
import os

# Load T=270 and T=290 for the same am/pwv and compare
f270 = "telluric_grid/am1.500_pwv2.000_T270.0_lam1500-2500_rp300000_cfgbbdc28a1dd31.dat"
f290 = "telluric_grid/am1.500_pwv2.000_T290.0_lam1500-2500_rp300000_cfg3a5eb0286df3.dat"

if os.path.exists(f270) and os.path.exists(f290):
    d270 = np.loadtxt(f270)
    d290 = np.loadtxt(f290)
    print("Max diff T270 vs T290:", np.max(np.abs(d270[:,1] - d290[:,1])))
    print("T270 transmission range:", d270[:,1].min(), d270[:,1].max())
    print("T290 transmission range:", d290[:,1].min(), d290[:,1].max())