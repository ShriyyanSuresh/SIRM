import glob
import re
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def revisit_index(path):
    m = re.search(r"-(\d+)-L2A_data\.tiff$", path)
    return int(m.group(1)) if m else 0

files = sorted(
    glob.glob("src/datasets/worldstrat/lr_dataset/ASMSpotter-1-3-1/L2A/*-L2A_data.tiff"),
    key=revisit_index,
)
files = files[:8]  # the first 8, correctly ordered now

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, (f, ax) in enumerate(zip(files, axes.flat)):
    with rasterio.open(f) as src:
        band0 = src.read(1)
    ax.imshow(band0, cmap="gray")
    ax.set_title(f"revisit {revisit_index(f)}\nstd={band0.std():.4f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("debug_all_revisits_band0.png", dpi=150)