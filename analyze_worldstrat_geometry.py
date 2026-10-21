import os
import ast
import json
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIG
# ============================================================

ROOT = r"data\worldstrat"

METADATA_CSV = os.path.join(ROOT, "metadata.csv")
HR_DIR = os.path.join(ROOT, "hr_dataset")
LR_DIR = os.path.join(ROOT, "lr_dataset")

OUTPUT = "worldstrat_geometry_report.json"


# ============================================================
# HELPERS
# ============================================================

def parse_bounds(value):
    """
    Convert metadata bounds string into:
    [min_lon, min_lat, max_lon, max_lat]
    """
    if isinstance(value, (list, tuple)):
        return list(value)

    try:
        return list(ast.literal_eval(value))
    except Exception:
        return None


def find_file(folder, suffix):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(suffix):
                return os.path.join(root, f)
    return None


def inspect_raster(path):
    with rasterio.open(path) as src:
        return {
            "path": path,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "resolution": list(src.res),
            "identity_transform": src.transform.is_identity,
            "bounds": list(src.bounds),
        }


# ============================================================
# STEP 1 — LOAD METADATA
# ============================================================

print("=" * 70)
print("STEP 1/6: Loading WorldStrat metadata")
print("=" * 70)

metadata = pd.read_csv(METADATA_CSV)

print(f"Metadata rows: {len(metadata):,}")
print(f"Unique tiles: {metadata['tile'].nunique():,}")

# Use first tile as representative example
sample_tile = metadata.iloc[0]["tile"]

sample_rows = metadata[metadata["tile"] == sample_tile].copy()

print(f"\nSample tile: {sample_tile}")
print(f"Observations: {len(sample_rows)}")

bounds = parse_bounds(sample_rows.iloc[0]["bounds"])

print(f"Bounds: {bounds}")


# ============================================================
# STEP 2 — FIND HR FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2/6: Inspecting HR products")
print("=" * 70)

tile_hr_dir = os.path.join(HR_DIR, sample_tile)

rgbn_path = find_file(tile_hr_dir, "_rgbn.tiff")
ps_path = find_file(tile_hr_dir, "_ps.tiff")
pan_path = find_file(tile_hr_dir, "_pan.tiff")

if not rgbn_path:
    raise FileNotFoundError("RGBN file not found")

if not ps_path:
    raise FileNotFoundError("PS file not found")

if not pan_path:
    raise FileNotFoundError("PAN file not found")


rgbn = inspect_raster(rgbn_path)
ps = inspect_raster(ps_path)
pan = inspect_raster(pan_path)

print("\nRGBN:")
print(json.dumps(rgbn, indent=2))

print("\nPS:")
print(json.dumps(ps, indent=2))

print("\nPAN:")
print(json.dumps(pan, indent=2))


# ============================================================
# STEP 3 — FIND LR FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3/6: Inspecting LR product")
print("=" * 70)

lr_tile_dir = os.path.join(LR_DIR, sample_tile)

# Look for first L2A observation
lr_path = None

for root, dirs, files in os.walk(lr_tile_dir):
    for f in files:
        if f.endswith("-L2A_data.tiff"):
            lr_path = os.path.join(root, f)
            break
    if lr_path:
        break

if not lr_path:
    raise FileNotFoundError("LR L2A file not found")

lr = inspect_raster(lr_path)

print(json.dumps(lr, indent=2))


# ============================================================
# STEP 4 — CALCULATE EFFECTIVE HR RESOLUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4/6: Calculating effective HR resolution")
print("=" * 70)

min_lon, min_lat, max_lon, max_lat = bounds

geo_width = max_lon - min_lon
geo_height = max_lat - min_lat

print(f"Geographic width : {geo_width:.10f} degrees")
print(f"Geographic height: {geo_height:.10f} degrees")


def calculate_effective_resolution(width, height):
    return {
        "x": geo_width / width,
        "y": geo_height / height,
    }


rgbn_res = calculate_effective_resolution(
    rgbn["width"],
    rgbn["height"]
)

ps_res = calculate_effective_resolution(
    ps["width"],
    ps["height"]
)

pan_res = calculate_effective_resolution(
    pan["width"],
    pan["height"]
)

print("\nEffective RGBN resolution:")
print(rgbn_res)

print("\nEffective PS resolution:")
print(ps_res)

print("\nEffective PAN resolution:")
print(pan_res)


# ============================================================
# STEP 5 — CALCULATE SCALE RATIOS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5/6: LR → HR scale analysis")
print("=" * 70)

lr_x = lr["resolution"][0]
lr_y = lr["resolution"][1]


def scale_ratio(hr_res):
    return {
        "x": lr_x / hr_res["x"],
        "y": lr_y / hr_res["y"],
    }


rgbn_scale = scale_ratio(rgbn_res)
ps_scale = scale_ratio(ps_res)
pan_scale = scale_ratio(pan_res)

print("\nLR → RGBN scale:")
print(rgbn_scale)

print("\nLR → PS scale:")
print(ps_scale)

print("\nLR → PAN scale:")
print(pan_scale)


# ============================================================
# STEP 6 — INSPECT .metadata FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 6/6: Inspecting WorldStrat .metadata files")
print("=" * 70)

metadata_files = []

for root, dirs, files in os.walk(lr_tile_dir):
    for f in files:
        if ".metadata" in f.lower():
            metadata_files.append(os.path.join(root, f))

print(f"Found {len(metadata_files)} .metadata files")

metadata_preview = None

if metadata_files:

    sample_metadata = metadata_files[0]

    print(f"\nSample metadata file:")
    print(sample_metadata)

    try:
        with open(
            sample_metadata,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            lines = f.readlines()

        metadata_preview = lines[:100]

        print("\n".join(
            line.rstrip()
            for line in metadata_preview
        ))

    except Exception as e:
        print(f"Could not read metadata file: {e}")


# ============================================================
# FINAL REPORT
# ============================================================

report = {

    "sample_tile": sample_tile,

    "metadata": {
        "rows": len(metadata),
        "unique_tiles": int(metadata["tile"].nunique()),
        "observations_for_sample_tile": len(sample_rows),
        "bounds": bounds,
    },

    "rgbn": rgbn,
    "ps": ps,
    "pan": pan,
    "lr": lr,

    "effective_hr_resolution": {
        "rgbn": rgbn_res,
        "ps": ps_res,
        "pan": pan_res,
    },

    "lr_to_hr_scale": {
        "rgbn": rgbn_scale,
        "ps": ps_scale,
        "pan": pan_scale,
    },

    "metadata_files": {
        "count": len(metadata_files),
        "sample": metadata_files[0] if metadata_files else None,
        "preview": metadata_preview,
    }
}


with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nReport saved to: {OUTPUT}")

print("\nThe important numbers are:")
print(f"RGBN scale: {rgbn_scale}")
print(f"PS scale  : {ps_scale}")
print(f"PAN scale : {pan_scale}")