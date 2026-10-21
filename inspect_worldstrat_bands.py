import os
import json
import re
from collections import Counter

import rasterio


# ============================================================
# CONFIG
# ============================================================

ROOT = r"data\worldstrat"

HR_DIR = os.path.join(ROOT, "hr_dataset")
LR_DIR = os.path.join(ROOT, "lr_dataset")

OUTPUT = "worldstrat_band_inspection.json"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def read_text_file(path, max_chars=30000):
    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            return f.read(max_chars)
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"


def inspect_raster(path):
    result = {
        "path": path
    }

    try:
        with rasterio.open(path) as src:

            result.update({
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": list(src.dtypes),
                "crs": str(src.crs),
                "descriptions": list(src.descriptions),
                "tags": dict(src.tags()),
                "transform": list(src.transform),
                "resolution": list(src.res),
                "nodata": src.nodata,
                "identity_transform": src.transform.is_identity,
            })

            band_info = []

            for i in range(1, src.count + 1):

                info = {
                    "band": i,
                    "description": src.descriptions[i - 1],
                    "tags": dict(src.tags(i)),
                }

                band_info.append(info)

            result["band_info"] = band_info

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# STEP 1 — FIND WORLDSTRAT DOCUMENTATION / METADATA FILES
# ============================================================

print_header("STEP 1/6: Searching for WorldStrat documentation")

candidate_files = []

for root, dirs, files in os.walk(ROOT):

    for file in files:

        lower = file.lower()

        if (
            lower.endswith(".txt")
            or lower.endswith(".md")
            or lower.endswith(".json")
            or lower.endswith(".xml")
            or lower.endswith(".csv")
            or lower.endswith(".yaml")
            or lower.endswith(".yml")
        ):
            candidate_files.append(
                os.path.join(root, file)
            )

print(f"Candidate documentation/metadata files: {len(candidate_files)}")

for path in candidate_files[:100]:
    print(path)


# ============================================================
# STEP 2 — SEARCH FILE CONTENT FOR BAND DEFINITIONS
# ============================================================

print_header("STEP 2/6: Searching for band definitions")

keywords = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
    "Sentinel",
    "RGBN",
    "PS",
    "PAN",
    "band",
    "spectral",
    "resolution",
]

matches = []

for path in candidate_files:

    # Avoid dumping huge files such as metadata.csv
    try:
        size = os.path.getsize(path)
    except Exception:
        continue

    if size > 10 * 1024 * 1024:
        continue

    content = read_text_file(path)

    found = []

    for keyword in keywords:

        if keyword.lower() in content.lower():
            found.append(keyword)

    if found:

        matches.append({
            "file": path,
            "keywords": found,
            "preview": content[:5000]
        })

        print(f"\nFOUND: {path}")
        print(f"Keywords: {found}")


# ============================================================
# STEP 3 — INSPECT A SAMPLE LR FILE
# ============================================================

print_header("STEP 3/6: Inspecting LR Sentinel-2 file")

sample_lr = None

for root, dirs, files in os.walk(LR_DIR):

    for file in files:

        if file.endswith("-L2A_data.tiff"):

            sample_lr = os.path.join(root, file)
            break

    if sample_lr:
        break


if not sample_lr:
    raise FileNotFoundError(
        "Could not find a -L2A_data.tiff file"
    )

print(f"Sample LR file:\n{sample_lr}")

lr_info = inspect_raster(sample_lr)

print(json.dumps(lr_info, indent=2))


# ============================================================
# STEP 4 — SEARCH LR FOLDER FOR SENTINEL BAND FILES
# ============================================================

print_header("STEP 4/6: Inspecting LR folder contents")

lr_folder = os.path.dirname(sample_lr)

lr_files = []

for file in os.listdir(lr_folder):

    path = os.path.join(lr_folder, file)

    if os.path.isfile(path):
        lr_files.append(file)

print(f"Files in sample LR observation folder: {len(lr_files)}")

for file in sorted(lr_files):

    print(file)


# ============================================================
# STEP 5 — INSPECT HR RGBN / PS / PAN
# ============================================================

print_header("STEP 5/6: Inspecting HR products")

# Get sample tile from the LR path.
#
# Structure:
# lr_dataset/
#   TILE/
#      L2A/
#
# Therefore parent of L2A = TILE.

tile_dir = os.path.dirname(lr_folder)
sample_tile = os.path.basename(tile_dir)

print(f"Sample tile: {sample_tile}")

hr_tile_dir = os.path.join(HR_DIR, sample_tile)

print(f"HR tile directory: {hr_tile_dir}")

if not os.path.exists(hr_tile_dir):

    raise FileNotFoundError(
        f"HR tile directory not found: {hr_tile_dir}"
    )


hr_files = []

for file in os.listdir(hr_tile_dir):

    path = os.path.join(hr_tile_dir, file)

    if os.path.isfile(path):
        hr_files.append(file)

print("\nHR files:")

for file in sorted(hr_files):
    print(file)


rgbn_path = None
ps_path = None
pan_path = None

for file in hr_files:

    lower = file.lower()

    if lower.endswith("_rgbn.tiff"):
        rgbn_path = os.path.join(hr_tile_dir, file)

    elif lower.endswith("_ps.tiff"):
        ps_path = os.path.join(hr_tile_dir, file)

    elif lower.endswith("_pan.tiff"):
        pan_path = os.path.join(hr_tile_dir, file)


hr_info = {}

if rgbn_path:
    print("\nRGBN:")
    hr_info["rgbn"] = inspect_raster(rgbn_path)
    print(json.dumps(hr_info["rgbn"], indent=2))

if ps_path:
    print("\nPS:")
    hr_info["ps"] = inspect_raster(ps_path)
    print(json.dumps(hr_info["ps"], indent=2))

if pan_path:
    print("\nPAN:")
    hr_info["pan"] = inspect_raster(pan_path)
    print(json.dumps(hr_info["pan"], indent=2))


# ============================================================
# STEP 6 — ANALYZE FILENAMES FOR BAND INFORMATION
# ============================================================

print_header("STEP 6/6: Searching filenames for band information")

all_filenames = []

for root, dirs, files in os.walk(LR_DIR):

    for file in files:
        all_filenames.append(file)


band_patterns = [
    r"B0[1-9]",
    r"B1[0-2]",
    r"B8A",
    r"RGBN",
    r"PAN",
    r"PS",
]


band_filename_matches = Counter()

examples = {}

for filename in all_filenames:

    for pattern in band_patterns:

        found = re.findall(
            pattern,
            filename,
            flags=re.IGNORECASE
        )

        for item in found:

            key = item.upper()

            band_filename_matches[key] += 1

            if key not in examples:
                examples[key] = filename


print("\nBand-related filename occurrences:")

for key, count in sorted(
    band_filename_matches.items()
):

    print(
        f"{key}: {count:,} "
        f"(example: {examples[key]})"
    )


# ============================================================
# FINAL REPORT
# ============================================================

report = {

    "sample_tile": sample_tile,

    "documentation_files": [
        {
            "file": x["file"],
            "keywords": x["keywords"],
            "preview": x["preview"]
        }
        for x in matches
    ],

    "sample_lr": lr_info,

    "sample_lr_folder": {
        "path": lr_folder,
        "files": sorted(lr_files)
    },

    "sample_hr": hr_info,

    "filename_band_analysis": {
        "counts": dict(band_filename_matches),
        "examples": examples
    }
}


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print_header("DONE")

print(f"Report saved to: {OUTPUT}")

print("\nIMPORTANT:")
print("Send me the terminal output from this script.")
print("We specifically need to see:")
print("1. Any B01-B12 information")
print("2. Any RGBN / PS / PAN documentation")
print("3. LR folder filenames")
print("4. Raster band descriptions/tags")