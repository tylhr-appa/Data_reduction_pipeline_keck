import os

# Find all .dat files under the current directory
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".dat"):
            print(os.path.join(root, f))

# Also check if OUT_DIR is being set to something unexpected
print("\nOUT_DIR =", "telluric_grid")
print("abs path =", os.path.abspath("telluric_grid"))