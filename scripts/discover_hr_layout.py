"""
Run this once to confirm the hr_dataset/ layout our dataset class assumes.

    python scripts/discover_hr_layout.py

If the printed structure doesn't match the pattern used in
src/datasets/worldstrat_dataset.py::_hr_ps_path, edit that one function -
nothing else needs to change.
"""
import glob
import os

ROOT = os.path.join("src","datasets", "worldstrat", "hr_dataset")

if not os.path.isdir(ROOT):
    print(f"{ROOT} not found - check you're running this from the SIRM project root.")
    raise SystemExit(1)

entries = sorted(os.listdir(ROOT))[:10]
print(f"First 10 entries directly under {ROOT}:")
for e in entries:
    full = os.path.join(ROOT, e)
    print(" ", e, "(dir)" if os.path.isdir(full) else "(file)")

# If they're directories (mirroring lr_dataset), show one tile's contents.
first_dir = next((e for e in entries if os.path.isdir(os.path.join(ROOT, e))), None)
if first_dir:
    print(f"\nContents of {first_dir}/:")
    for f in sorted(os.listdir(os.path.join(ROOT, first_dir))):
        print(" ", f)
else:
    print("\nNo subdirectories found - hr_dataset appears to be a flat file layout.")
    print("Sample filenames matching *ps*:")
    for f in glob.glob(os.path.join(ROOT, "*ps*"))[:5]:
        print(" ", f)
