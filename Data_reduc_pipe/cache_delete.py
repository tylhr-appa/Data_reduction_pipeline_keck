import os, glob, shutil

# HDF5 grid files
for h5 in glob.glob("*.h5"):
    os.remove(h5)
    print(f"Deleted {h5}")

# Cached PSG .dat files in telluric_grid/
dat_files = glob.glob("telluric_grid/*.dat")
for f in dat_files:
    os.remove(f)
print(f"Deleted {len(dat_files)} .dat files from telluric_grid/")
