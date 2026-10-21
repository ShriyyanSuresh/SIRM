import os
import re
import json
import statistics
from collections import defaultdict

import rasterio
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = "data/worldstrat"

HR_DIR = os.path.join(ROOT, "hr_dataset")
LR_DIR = os.path.join(ROOT, "lr_dataset")

METADATA_CSV = os.path.join(ROOT, "metadata.csv")
SPLIT_CSV = os.path.join(ROOT, "stratified_train_val_test_split.csv")

OUTPUT_JSON = "worldstrat_v2_report.json"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_files(root, suffix):
    results = []

    if not os.path.exists(root):
        return results

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(suffix.lower()):
                results.append(os.path.join(dirpath, filename))

    return results


def get_tile_id_hr(filename):
    """
    Examples:

    ASMSpotter-1-1-1_rgbn.tiff
    ASMSpotter-1-1-1_ps.tiff
    ASMSpotter-1-1-1_pan.tiff

    -> ASMSpotter-1-1-1
    """

    name = os.path.basename(filename)

    for suffix in ["_rgbn.tiff", "_ps.tiff", "_pan.tiff"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return None


def get_tile_id_lr(filename):
    """
    Example:

    ASMSpotter-1-1-1-12-L2A_data.tiff

    -> ASMSpotter-1-1-1

    The final -12 is the observation number.
    """

    name = os.path.basename(filename)

    match = re.match(
        r"^(.*)-(\d+)-L2A_data\.tiff$",
        name,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def get_observation_number(filename):
    name = os.path.basename(filename)

    match = re.search(
        r"-(\d+)-L2A_data\.tiff$",
        name,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def raster_summary(path):
    try:
        with rasterio.open(path) as src:

            transform = src.transform

            return {
                "path": path,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs),
                "resolution": list(src.res),
                "bounds": list(src.bounds),
                "transform": list(transform),
                "identity_transform": transform == rasterio.Affine.identity(),
                "nodata": src.nodata,
                "band_descriptions": list(src.descriptions),
            }

    except Exception as e:

        return {
            "path": path,
            "error": str(e)
        }


# ============================================================
# STEP 1
# FIND HR FILES
# ============================================================

print_header("STEP 1/8: FINDING HR FILES")

rgbn_files = find_files(HR_DIR, "_rgbn.tiff")
ps_files = find_files(HR_DIR, "_ps.tiff")
pan_files = find_files(HR_DIR, "_pan.tiff")

print(f"HR RGBN files: {len(rgbn_files)}")
print(f"HR PS files:   {len(ps_files)}")
print(f"HR PAN files:  {len(pan_files)}")


# ============================================================
# STEP 2
# FIND LR FILES
# ============================================================

print_header("STEP 2/8: FINDING LR L2A FILES")

lr_files = find_files(LR_DIR, "-L2A_data.tiff")

print(f"LR L2A observations: {len(lr_files)}")


# ============================================================
# STEP 3
# INDEX HR FILES
# ============================================================

print_header("STEP 3/8: INDEXING HR DATA")

hr_rgbn = {}
hr_ps = {}
hr_pan = {}

for path in rgbn_files:
    tile = get_tile_id_hr(path)

    if tile:
        hr_rgbn[tile] = path

for path in ps_files:
    tile = get_tile_id_hr(path)

    if tile:
        hr_ps[tile] = path

for path in pan_files:
    tile = get_tile_id_hr(path)

    if tile:
        hr_pan[tile] = path


print(f"Unique RGBN tiles: {len(hr_rgbn)}")
print(f"Unique PS tiles:   {len(hr_ps)}")
print(f"Unique PAN tiles:  {len(hr_pan)}")


# ============================================================
# STEP 4
# INDEX LR OBSERVATIONS
# ============================================================

print_header("STEP 4/8: INDEXING LR OBSERVATIONS")

lr_by_tile = defaultdict(list)

for path in lr_files:

    tile = get_tile_id_lr(path)
    obs = get_observation_number(path)

    if tile:

        lr_by_tile[tile].append({
            "path": path,
            "observation": obs
        })


observation_counts = [
    len(v)
    for v in lr_by_tile.values()
]

print(f"Unique LR tiles: {len(lr_by_tile)}")

if observation_counts:

    print(
        f"Observations per tile: "
        f"min={min(observation_counts)}, "
        f"max={max(observation_counts)}, "
        f"mean={statistics.mean(observation_counts):.2f}, "
        f"median={statistics.median(observation_counts)}"
    )


# ============================================================
# STEP 5
# INSPECT HR FORMATS
# ============================================================

print_header("STEP 5/8: INSPECTING HR FORMATS")

hr_samples = {}

if rgbn_files:
    path = rgbn_files[0]

    print("\nRGBN SAMPLE")
    print("-" * 50)

    summary = raster_summary(path)
    print(json.dumps(summary, indent=2))

    hr_samples["rgbn"] = summary


if ps_files:
    path = ps_files[0]

    print("\nPS SAMPLE")
    print("-" * 50)

    summary = raster_summary(path)
    print(json.dumps(summary, indent=2))

    hr_samples["ps"] = summary


if pan_files:
    path = pan_files[0]

    print("\nPAN SAMPLE")
    print("-" * 50)

    summary = raster_summary(path)
    print(json.dumps(summary, indent=2))

    hr_samples["pan"] = summary


# ============================================================
# STEP 6
# INSPECT LR DATA + BAND INFORMATION
# ============================================================

print_header("STEP 6/8: INSPECTING LR 12-BAND DATA")

lr_samples = []

sample_lr = sorted(lr_files)[:3]

for path in sample_lr:

    print("\nLR SAMPLE")
    print("-" * 50)

    summary = raster_summary(path)

    print(json.dumps(summary, indent=2))

    try:

        with rasterio.open(path) as src:

            print("\nBand information:")

            for i in range(1, src.count + 1):

                description = src.descriptions[i - 1]

                tags = src.tags(i)

                print(
                    f"Band {i}: "
                    f"description={description}, "
                    f"tags={tags}"
                )

    except Exception as e:

        print(f"Could not inspect bands: {e}")

    lr_samples.append(summary)


# ============================================================
# STEP 7
# CHECK METADATA + DATES + SPLITS
# ============================================================

print_header("STEP 7/8: METADATA / DATES / SPLITS")

metadata_info = {}

if os.path.exists(METADATA_CSV):

    metadata = pd.read_csv(METADATA_CSV)

    print("\nMetadata:")
    print(f"Shape: {metadata.shape}")

    print("\nColumns:")
    print(list(metadata.columns))

    print("\nFirst 5 rows:")
    print(metadata.head().to_string())

    metadata_info["shape"] = list(metadata.shape)
    metadata_info["columns"] = list(metadata.columns)

    # --------------------------------------------------------
    # DATE ANALYSIS
    # --------------------------------------------------------

    if "lowres_date" in metadata.columns:

        metadata["lowres_date_parsed"] = pd.to_datetime(
            metadata["lowres_date"],
            errors="coerce"
        )

        print("\nLow-resolution dates:")

        print(
            f"Min: {metadata['lowres_date_parsed'].min()}"
        )

        print(
            f"Max: {metadata['lowres_date_parsed'].max()}"
        )

    if "highres_date" in metadata.columns:

        metadata["highres_date_parsed"] = pd.to_datetime(
            metadata["highres_date"],
            errors="coerce"
        )

        print("\nHigh-resolution dates:")

        print(
            f"Min: {metadata['highres_date_parsed'].min()}"
        )

        print(
            f"Max: {metadata['highres_date_parsed'].max()}"
        )

    # --------------------------------------------------------
    # TILE COUNTS
    # --------------------------------------------------------

    if "tile" in metadata.columns:

        print(
            f"\nUnique metadata tiles: "
            f"{metadata['tile'].nunique()}"
        )

        print(
            f"Metadata rows: "
            f"{len(metadata)}"
        )

        counts = metadata["tile"].value_counts()

        print("\nMetadata rows per tile:")
        print(
            f"min={counts.min()}, "
            f"max={counts.max()}, "
            f"mean={counts.mean():.2f}, "
            f"median={counts.median()}"
        )

else:

    print("metadata.csv NOT FOUND")


# ------------------------------------------------------------
# SPLIT FILE
# ------------------------------------------------------------

split_info = {}

if os.path.exists(SPLIT_CSV):

    split = pd.read_csv(SPLIT_CSV)

    print("\nSplit CSV:")
    print(f"Shape: {split.shape}")

    print("\nColumns:")
    print(list(split.columns))

    if "split" in split.columns:

        print("\nSplit distribution:")
        print(split["split"].value_counts())

    if "tile" in split.columns:

        duplicates = split[
            split["tile"].duplicated(keep=False)
        ]

        print(
            f"\nDuplicate tile rows: "
            f"{len(duplicates)}"
        )

        conflicting = []

        for tile, group in split.groupby("tile"):

            if group["split"].nunique() > 1:

                conflicting.append(tile)

        print(
            f"Tiles with conflicting splits: "
            f"{len(conflicting)}"
        )

        if conflicting:

            print("\nExample conflicting tiles:")

            for tile in conflicting[:10]:

                print(
                    split[
                        split["tile"] == tile
                    ].to_string(index=False)
                )

    split_info["shape"] = list(split.shape)

else:

    print("Split CSV NOT FOUND")


# ============================================================
# STEP 8
# CREATE TRAINING-PAIR DIAGNOSTICS
# ============================================================

print_header("STEP 8/8: TRAINING-PAIR DIAGNOSTICS")

common_tiles = (
    set(hr_rgbn.keys())
    & set(lr_by_tile.keys())
)

print(f"RGBN + LR common tiles: {len(common_tiles)}")

print(
    f"RGBN tiles without LR: "
    f"{len(set(hr_rgbn.keys()) - set(lr_by_tile.keys()))}"
)

print(
    f"LR tiles without RGBN: "
    f"{len(set(lr_by_tile.keys()) - set(hr_rgbn.keys()))}"
)

print("\nExample training pair:")

if common_tiles:

    tile = sorted(common_tiles)[0]

    hr_path = hr_rgbn[tile]
    lr_path = lr_by_tile[tile][0]["path"]

    print(f"\nTile: {tile}")

    print(f"HR RGBN:")
    print(f"  {hr_path}")

    print(f"\nLR:")
    print(f"  {lr_path}")

    hr_info = raster_summary(hr_path)
    lr_info = raster_summary(lr_path)

    print("\nHR:")
    print(
        f"  Size: "
        f"{hr_info.get('width')} × "
        f"{hr_info.get('height')}"
    )

    print(
        f"  Bands: "
        f"{hr_info.get('bands')}"
    )

    print(
        f"  CRS: "
        f"{hr_info.get('crs')}"
    )

    print(
        f"  Resolution: "
        f"{hr_info.get('resolution')}"
    )

    print(
        f"  Identity transform: "
        f"{hr_info.get('identity_transform')}"
    )

    print("\nLR:")

    print(
        f"  Size: "
        f"{lr_info.get('width')} × "
        f"{lr_info.get('height')}"
    )

    print(
        f"  Bands: "
        f"{lr_info.get('bands')}"
    )

    print(
        f"  CRS: "
        f"{lr_info.get('crs')}"
    )

    print(
        f"  Resolution: "
        f"{lr_info.get('resolution')}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT calculate geographic scale if HR transform
    # is identity.
    # --------------------------------------------------------

    if (
        not hr_info.get("identity_transform")
        and hr_info.get("resolution")
        and lr_info.get("resolution")
    ):

        x_ratio = (
            lr_info["resolution"][0]
            / hr_info["resolution"][0]
        )

        y_ratio = (
            lr_info["resolution"][1]
            / hr_info["resolution"][1]
        )

        print("\nEstimated LR/HR resolution ratio:")

        print(f"  X: {x_ratio}")
        print(f"  Y: {y_ratio}")

    else:

        print(
            "\nTrue geographic resolution ratio:"
        )

        print(
            "  CANNOT be calculated from HR TIFF metadata."
        )

        print(
            "  HR has an identity transform."
        )


# ============================================================
# REPORT
# ============================================================

report = {

    "hr_counts": {
        "rgbn": len(rgbn_files),
        "ps": len(ps_files),
        "pan": len(pan_files)
    },

    "lr_count": len(lr_files),

    "unique_lr_tiles": len(lr_by_tile),

    "observation_statistics": {
        "min": min(observation_counts)
        if observation_counts else None,

        "max": max(observation_counts)
        if observation_counts else None,

        "mean": statistics.mean(observation_counts)
        if observation_counts else None,

        "median": statistics.median(observation_counts)
        if observation_counts else None
    },

    "hr_samples": hr_samples,

    "lr_samples": lr_samples,

    "metadata": metadata_info,

    "split": split_info,

    "common_rgbn_lr_tiles": len(common_tiles)
}


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        report,
        f,
        indent=2,
        default=str
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL SUMMARY")

print(f"HR RGBN tiles:       {len(rgbn_files)}")
print(f"HR PS tiles:         {len(ps_files)}")
print(f"HR PAN tiles:        {len(pan_files)}")
print(f"LR observations:     {len(lr_files)}")
print(f"Unique LR tiles:     {len(lr_by_tile)}")
print(f"Common RGBN/LR:      {len(common_tiles)}")

print("\nReport written to:")
print(OUTPUT_JSON)

print("\nNext step:")
print(
    "Send me the output from STEP 5, STEP 6, "
    "STEP 7 and STEP 8."
)