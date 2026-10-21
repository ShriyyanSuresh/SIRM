"""
WorldStrat dataset inspection script for SIRM.

Run from the SIRM project root:
    python inspect_worldstrat.py

Optional:
    python inspect_worldstrat.py --full
    python inspect_worldstrat.py --sample 500
    python inspect_worldstrat.py --root path/to/worldstrat

Outputs:
    worldstrat_inspection_report.json

This script:
1. Locates HR and LR WorldStrat files.
2. Inspects metadata.csv and the train/val/test split.
3. Inspects representative HR/LR imagery.
4. Pairs REAL HR/LR tiles using their POI/tile IDs.
5. Checks HR/LR spatial/geometric compatibility.
6. Checks for unreadable/corrupt TIFF files.

Important:
WorldStrat has:
    hr_dataset/
        <TILE>/
            <TILE>_rgbn.tiff
            <TILE>_ps.tiff
            <TILE>_pan.tiff

    lr_dataset/
        <TILE>/
            L2A/
                <TILE>-<N>-L2A_data.tiff
                <TILE>-<N>-CLM.tiff
                ...

For SIRM we primarily care about:
    HR: *_rgbn.tiff
    LR: *-L2A_data.tiff
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import traceback
from collections import Counter, defaultdict

import pandas as pd

try:
    import rasterio
except ImportError:
    print("rasterio is not installed.")
    print("Run: pip install rasterio")
    sys.exit(1)


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_tif_files(root_dir):
    """Find every TIFF recursively."""
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(root_dir, "**", pattern),
                recursive=True
            )
        )

    return sorted(set(files))


def find_hr_rgbn_files(hr_dir):
    """Find HR RGBN images, which are the main HR target candidates."""
    patterns = ["*_rgbn.tiff", "*_rgbn.tif"]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(hr_dir, "**", pattern),
                recursive=True
            )
        )

    return sorted(set(files))


def find_hr_ps_files(hr_dir):
    """Find HR multispectral/pansharpened PS images."""
    patterns = ["*_ps.tiff", "*_ps.tif"]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(hr_dir, "**", pattern),
                recursive=True
            )
        )

    return sorted(set(files))


def find_hr_pan_files(hr_dir):
    """Find HR panchromatic images."""
    patterns = ["*_pan.tiff", "*_pan.tif"]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(hr_dir, "**", pattern),
                recursive=True
            )
        )

    return sorted(set(files))


def find_lr_l2a_data_files(lr_dir):
    """
    Find only actual Sentinel-2 L2A image data.

    We intentionally exclude:
        CLM
        CLP
        dataMask
        sunAzimuthAngles
        sunZenithAngles
        viewAzimuthAngles
        viewZenithAngles
    """

    patterns = [
        "*-L2A_data.tiff",
        "*-L2A_data.tif",
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(lr_dir, "**", pattern),
                recursive=True
            )
        )

    return sorted(set(files))


# ============================================================
# TILE ID EXTRACTION
# ============================================================

def extract_tile_from_hr(path):
    """
    Example:

        data/worldstrat/hr_dataset/
        Amnesty POI-1-1-1/
        Amnesty POI-1-1-1_rgbn.tiff

    Returns:

        Amnesty POI-1-1-1
    """

    filename = os.path.basename(path)

    suffixes = [
        "_rgbn.tiff",
        "_rgbn.tif",
        "_ps.tiff",
        "_ps.tif",
        "_pan.tiff",
        "_pan.tif",
    ]

    for suffix in suffixes:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]

    return os.path.splitext(filename)[0]


def extract_tile_from_lr(path):
    """
    Example:

        Amnesty POI-1-1-1-17-L2A_data.tiff

    Returns:

        Amnesty POI-1-1-1

    The final '-N' is the LR acquisition/sample number.
    """

    filename = os.path.basename(path)

    filename = re.sub(
        r"-L2A_data\.(tiff|tif)$",
        "",
        filename,
        flags=re.IGNORECASE
    )

    # Remove final acquisition number.
    #
    # Example:
    # Amnesty POI-1-1-1-17
    #
    # becomes:
    # Amnesty POI-1-1-1

    match = re.match(r"^(.*)-(\d+)$", filename)

    if match:
        return match.group(1)

    return filename


def extract_lr_observation_number(path):
    """
    Example:

        Amnesty POI-1-1-1-17-L2A_data.tiff

    Returns:

        17
    """

    filename = os.path.basename(path)

    match = re.search(
        r"-(\d+)-L2A_data\.(tiff|tif)$",
        filename,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# TIFF INSPECTION
# ============================================================

def safe_tif_summary(path):
    """
    Open a GeoTIFF and extract structural information.
    """

    out = {
        "path": path,
        "ok": True,
        "error": None,
    }

    try:

        with rasterio.open(path) as src:

            out.update({

                "width": src.width,

                "height": src.height,

                "count": src.count,

                "dtype": src.dtypes[0] if src.count else None,

                "all_dtypes": list(src.dtypes),

                "crs": str(src.crs) if src.crs else None,

                "transform": list(src.transform)[:6],

                "res": list(src.res),

                "bounds": list(src.bounds),

                "nodata": src.nodata,

                "band_descriptions": list(src.descriptions),

                "tags": dict(src.tags()),

            })

            # ------------------------------------------------
            # Pixel statistics
            # ------------------------------------------------

            stats = []

            for band_index in range(1, src.count + 1):

                try:

                    band = src.read(
                        band_index,
                        masked=True
                    )

                    if band.size:

                        stats.append({

                            "band": band_index,

                            "min": float(band.min()),

                            "max": float(band.max()),

                            "mean": float(band.mean()),

                            "std": float(band.std()),

                        })

                except Exception as e:

                    stats.append({

                        "band": band_index,

                        "error": str(e),

                    })

            out["band_statistics"] = stats

    except Exception as e:

        out["ok"] = False

        out["error"] = f"{type(e).__name__}: {e}"

    return out


# ============================================================
# HR/LR PAIRING
# ============================================================

def build_hr_index(hr_rgbn_files):
    """
    Build:

        TILE -> HR RGBN path
    """

    index = {}

    for path in hr_rgbn_files:

        tile = extract_tile_from_hr(path)

        index[tile] = path

    return index


def build_lr_index(lr_l2a_files):
    """
    Build:

        TILE -> list of LR observations
    """

    index = defaultdict(list)

    for path in lr_l2a_files:

        tile = extract_tile_from_lr(path)

        index[tile].append(path)

    return dict(index)


def find_matching_pairs(hr_rgbn_files, lr_l2a_files):
    """
    Match HR RGBN tiles with all LR L2A observations belonging
    to the same tile.
    """

    hr_index = build_hr_index(hr_rgbn_files)

    lr_index = build_lr_index(lr_l2a_files)

    pairs = []

    for tile, hr_path in sorted(hr_index.items()):

        lr_paths = lr_index.get(tile, [])

        for lr_path in sorted(lr_paths):

            pairs.append({

                "tile": tile,

                "hr": hr_path,

                "lr": lr_path,

                "lr_observation":
                    extract_lr_observation_number(lr_path),

            })

    return pairs, hr_index, lr_index


# ============================================================
# ALIGNMENT
# ============================================================

def check_alignment(hr_summary, lr_summary):
    """
    Compare an actual matched HR/LR pair.

    Note:
    HR files in this WorldStrat distribution may not carry a
    meaningful geotransform. Therefore this function reports
    both geometric metadata and whether a strict spatial
    comparison is possible.
    """

    result = {}

    if not (
        hr_summary["ok"]
        and lr_summary["ok"]
    ):

        result["checked"] = False

        result["reason"] = (
            "one or both files failed to open"
        )

        return result

    result["checked"] = True

    result["same_crs"] = (
        hr_summary["crs"]
        == lr_summary["crs"]
    )

    result["hr_crs"] = hr_summary["crs"]

    result["lr_crs"] = lr_summary["crs"]

    result["hr_resolution"] = hr_summary["res"]

    result["lr_resolution"] = lr_summary["res"]

    result["hr_dimensions"] = [
        hr_summary["width"],
        hr_summary["height"],
    ]

    result["lr_dimensions"] = [
        lr_summary["width"],
        lr_summary["height"],
    ]

    # --------------------------------------------------------
    # Dimension scale
    # --------------------------------------------------------

    if (
        lr_summary["width"]
        and lr_summary["height"]
        and hr_summary["width"]
        and hr_summary["height"]
    ):

        result["width_scale_ratio"] = (
            hr_summary["width"]
            / lr_summary["width"]
        )

        result["height_scale_ratio"] = (
            hr_summary["height"]
            / lr_summary["height"]
        )

    else:

        result["width_scale_ratio"] = None

        result["height_scale_ratio"] = None

    # --------------------------------------------------------
    # Physical resolution
    # --------------------------------------------------------

    hr_res_x = hr_summary["res"][0]

    lr_res_x = lr_summary["res"][0]

    if hr_res_x and lr_res_x:

        result["resolution_ratio_lr_over_hr"] = (
            lr_res_x / hr_res_x
        )

    else:

        result["resolution_ratio_lr_over_hr"] = None

    # --------------------------------------------------------
    # Check georeferencing quality
    # --------------------------------------------------------

    hr_transform = hr_summary["transform"]

    lr_transform = lr_summary["transform"]

    hr_identity = (
        hr_transform == [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    result["hr_identity_transform"] = hr_identity

    result["lr_transform_available"] = (
        lr_transform is not None
    )

    if hr_identity:

        result["spatial_alignment_check"] = "not_reliable"

        result["spatial_alignment_reason"] = (
            "HR file has an identity transform, so its "
            "geographic pixel grid cannot be reliably compared "
            "with the LR geographic grid using the TIFF metadata."
        )

    else:

        result["spatial_alignment_check"] = "metadata_available"

    return result


# ============================================================
# CSV INSPECTION
# ============================================================

def inspect_csvs(root):

    meta_path = os.path.join(
        root,
        "metadata.csv"
    )

    split_path = os.path.join(
        root,
        "stratified_train_val_test_split.csv"
    )

    csv_report = {}

    print()
    print("=" * 70)
    print("STEP 2/7: Metadata + split CSV structure")
    print("=" * 70)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if os.path.exists(meta_path):

        try:

            df = pd.read_csv(meta_path)

            print(
                f"\nmetadata.csv -> shape={df.shape}"
            )

            print(
                "columns:",
                list(df.columns)
            )

            print(
                df.head(3).to_string()
            )

            csv_report["metadata"] = {

                "shape": list(df.shape),

                "columns": list(df.columns),

                "dtypes": {
                    c: str(t)
                    for c, t in df.dtypes.items()
                },

            }

        except Exception as e:

            print(
                "ERROR reading metadata.csv:",
                e
            )

            csv_report["metadata"] = {
                "error": str(e)
            }

    else:

        print(
            f"WARNING: {meta_path} not found."
        )

        csv_report["metadata"] = None

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    if os.path.exists(split_path):

        try:

            df = pd.read_csv(split_path)

            print(
                f"\nsplit.csv -> shape={df.shape}"
            )

            print(
                "columns:",
                list(df.columns)
            )

            print(
                df.head(3).to_string()
            )

            id_col = "tile" if "tile" in df.columns else df.columns[0]

            dup_mask = df.duplicated(
                subset=[id_col],
                keep=False
            )

            dupes = df[dup_mask]

            print(
                f"\nDuplicate rows on '{id_col}': "
                f"{len(dupes)}"
            )

            conflict_report = {}

            if len(dupes):

                print(
                    dupes
                    .sort_values(id_col)
                    .head(30)
                    .to_string()
                )

                # ------------------------------------------------
                # Detect conflicting split assignments
                # ------------------------------------------------

                if "split" in df.columns:

                    split_conflicts = (
                        dupes
                        .groupby(id_col)["split"]
                        .nunique()
                    )

                    conflicting_ids = (
                        split_conflicts[
                            split_conflicts > 1
                        ]
                    )

                    conflict_report[
                        "split"
                    ] = len(conflicting_ids)

                    print(
                        "\nTiles appearing in multiple "
                        "different splits:",
                        len(conflicting_ids)
                    )

                    if len(conflicting_ids):

                        print(
                            conflicting_ids
                            .head(30)
                            .to_string()
                        )

            csv_report["split"] = {

                "shape": list(df.shape),

                "columns": list(df.columns),

                "dtypes": {
                    c: str(t)
                    for c, t in df.dtypes.items()
                },

                "duplicate_tile_rows":
                    int(len(dupes)),

                "duplicate_conflicts":
                    conflict_report,

            }

        except Exception as e:

            print(
                "ERROR reading split CSV:",
                e
            )

            csv_report["split"] = {
                "error": str(e)
            }

    else:

        print(
            f"WARNING: {split_path} not found."
        )

        csv_report["split"] = None

    return csv_report


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_pair_statistics(pairs, hr_index, lr_index):

    print()
    print("=" * 70)
    print("PAIRING STATISTICS")
    print("=" * 70)

    print(
        "HR RGBN tiles:",
        len(hr_index)
    )

    print(
        "LR tiles:",
        len(lr_index)
    )

    print(
        "Matched HR/LR observations:",
        len(pairs)
    )

    hr_without_lr = sorted(
        set(hr_index)
        - set(lr_index)
    )

    lr_without_hr = sorted(
        set(lr_index)
        - set(hr_index)
    )

    print(
        "HR tiles without LR:",
        len(hr_without_lr)
    )

    print(
        "LR tiles without HR:",
        len(lr_without_hr)
    )

    observations_per_tile = [
        len(paths)
        for paths in lr_index.values()
    ]

    if observations_per_tile:

        print(
            "\nLR observations per tile:"
        )

        print(
            "  min:",
            min(observations_per_tile)
        )

        print(
            "  max:",
            max(observations_per_tile)
        )

        print(
            "  mean:",
            round(
                sum(observations_per_tile)
                / len(observations_per_tile),
                2
            )
        )

        print(
            "  median:",
            sorted(observations_per_tile)[
                len(observations_per_tile) // 2
            ]
        )

    print(
        "\nSample REAL pairs:"
    )

    for pair in pairs[:10]:

        print(
            f"\n  TILE: {pair['tile']}"
        )

        print(
            f"  HR:   {pair['hr']}"
        )

        print(
            f"  LR:   {pair['lr']}"
        )

        print(
            f"  LR observation: "
            f"{pair['lr_observation']}"
        )

    return {

        "hr_rgbn_tiles":
            len(hr_index),

        "lr_tiles":
            len(lr_index),

        "matched_observations":
            len(pairs),

        "hr_without_lr":
            len(hr_without_lr),

        "lr_without_hr":
            len(lr_without_hr),

        "lr_observations_per_tile": {

            "min":
                min(observations_per_tile)
                if observations_per_tile
                else None,

            "max":
                max(observations_per_tile)
                if observations_per_tile
                else None,

            "mean":
                (
                    sum(observations_per_tile)
                    / len(observations_per_tile)
                )
                if observations_per_tile
                else None,

            "median":
                (
                    sorted(observations_per_tile)
                    [len(observations_per_tile) // 2]
                )
                if observations_per_tile
                else None,

        },

    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=os.path.join(
            "data",
            "worldstrat"
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "scan every TIFF for readability "
            "instead of a sample"
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help=(
            "number of TIFFs to check in quick mode"
        )
    )

    args = parser.parse_args()

    root = args.root

    hr_dir = os.path.join(
        root,
        "hr_dataset"
    )

    lr_dir = os.path.join(
        root,
        "lr_dataset"
    )

    report = {}

    # ========================================================
    # STEP 1
    # ========================================================

    print("=" * 70)
    print("STEP 1/7: Locating WorldStrat files")
    print("=" * 70)

    if not os.path.isdir(hr_dir):

        print(
            f"WARNING: {hr_dir} not found."
        )

    if not os.path.isdir(lr_dir):

        print(
            f"WARNING: {lr_dir} not found."
        )

    # --------------------------------------------------------
    # All TIFFs
    # --------------------------------------------------------

    hr_files = (
        find_tif_files(hr_dir)
        if os.path.isdir(hr_dir)
        else []
    )

    lr_files = (
        find_tif_files(lr_dir)
        if os.path.isdir(lr_dir)
        else []
    )

    # --------------------------------------------------------
    # Important subsets
    # --------------------------------------------------------

    hr_rgbn_files = (
        find_hr_rgbn_files(hr_dir)
        if os.path.isdir(hr_dir)
        else []
    )

    hr_ps_files = (
        find_hr_ps_files(hr_dir)
        if os.path.isdir(hr_dir)
        else []
    )

    hr_pan_files = (
        find_hr_pan_files(hr_dir)
        if os.path.isdir(hr_dir)
        else []
    )

    lr_l2a_files = (
        find_lr_l2a_data_files(lr_dir)
        if os.path.isdir(lr_dir)
        else []
    )

    print(
        f"\nAll HR GeoTIFFs: {len(hr_files)}"
    )

    print(
        f"All LR GeoTIFFs: {len(lr_files)}"
    )

    print(
        f"\nHR RGBN files: {len(hr_rgbn_files)}"
    )

    print(
        f"HR PS files: {len(hr_ps_files)}"
    )

    print(
        f"HR PAN files: {len(hr_pan_files)}"
    )

    print(
        f"LR L2A_data files: {len(lr_l2a_files)}"
    )

    print(
        "\nSample HR RGBN paths:"
    )

    for f in hr_rgbn_files[:5]:

        print(
            " ",
            f
        )

    print(
        "\nSample LR L2A_data paths:"
    )

    for f in lr_l2a_files[:5]:

        print(
            " ",
            f
        )

    report["counts"] = {

        "all_hr_files":
            len(hr_files),

        "all_lr_files":
            len(lr_files),

        "hr_rgbn":
            len(hr_rgbn_files),

        "hr_ps":
            len(hr_ps_files),

        "hr_pan":
            len(hr_pan_files),

        "lr_l2a_data":
            len(lr_l2a_files),

    }

    # ========================================================
    # STEP 2
    # ========================================================

    csv_report = inspect_csvs(root)

    report["csv"] = csv_report

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3/7: Structural inspection")
    print("=" * 70)

    # --------------------------------------------------------
    # Pick representative files
    # --------------------------------------------------------

    # Use first files rather than random files so results are
    # reproducible.

    hr_samples = (
        hr_rgbn_files[:3]
        + hr_ps_files[:1]
        + hr_pan_files[:1]
    )

    lr_samples = lr_l2a_files[:5]

    hr_summaries = [
        safe_tif_summary(f)
        for f in hr_samples
    ]

    lr_summaries = [
        safe_tif_summary(f)
        for f in lr_samples
    ]

    print(
        "\n--- HR samples ---"
    )

    for summary in hr_summaries:

        print(
            json.dumps(
                summary,
                indent=2,
                default=str
            )
        )

    print(
        "\n--- LR L2A_data samples ---"
    )

    for summary in lr_summaries:

        print(
            json.dumps(
                summary,
                indent=2,
                default=str
            )
        )

    report["structural_samples"] = {

        "hr":
            hr_summaries,

        "lr_l2a_data":
            lr_summaries,

    }

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4/7: REAL HR/LR tile pairing")
    print("=" * 70)

    pairs, hr_index, lr_index = find_matching_pairs(
        hr_rgbn_files,
        lr_l2a_files
    )

    pair_statistics = print_pair_statistics(
        pairs,
        hr_index,
        lr_index
    )

    report["pairing"] = pair_statistics

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5/7: Actual HR/LR compatibility check")
    print("=" * 70)

    compatibility_report = []

    # Inspect first 10 actual matched pairs.

    for pair in pairs[:10]:

        print(
            f"\nTile: {pair['tile']}"
        )

        print(
            f"LR observation: "
            f"{pair['lr_observation']}"
        )

        print(
            "HR:"
        )

        print(
            f"  {pair['hr']}"
        )

        print(
            "LR:"
        )

        print(
            f"  {pair['lr']}"
        )

        hr_summary = safe_tif_summary(
            pair["hr"]
        )

        lr_summary = safe_tif_summary(
            pair["lr"]
        )

        alignment = check_alignment(
            hr_summary,
            lr_summary
        )

        print(
            "\nCompatibility:"
        )

        print(
            json.dumps(
                alignment,
                indent=2,
                default=str
            )
        )

        compatibility_report.append({

            "tile":
                pair["tile"],

            "lr_observation":
                pair["lr_observation"],

            "hr":
                pair["hr"],

            "lr":
                pair["lr"],

            "hr_summary":
                hr_summary,

            "lr_summary":
                lr_summary,

            "compatibility":
                alignment,

        })

    report["compatibility_samples"] = (
        compatibility_report
    )

    # ========================================================
    # STEP 6
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 6/7: Missing/corrupted file scan")
    print("=" * 70)

    all_files = (
        hr_files
        + lr_files
    )

    if args.full:

        scan_files = all_files

        print(
            "Full scan requested:"
        )

        print(
            f"Checking all {len(scan_files)} files."
        )

        print(
            "This may take a while."
        )

    else:

        k = min(
            args.sample,
            len(all_files)
        )

        if all_files:

            random.seed(42)

            scan_files = random.sample(
                all_files,
                k
            )

        else:

            scan_files = []

        print(
            f"Quick scan: checking "
            f"{len(scan_files)} of "
            f"{len(all_files)} files."
        )

        print(
            "Use --full for exhaustive checking."
        )

    bad_files = []

    for i, path in enumerate(
        scan_files
    ):

        try:

            with rasterio.open(path) as src:

                _ = (
                    src.width,
                    src.height,
                    src.count
                )

        except Exception as e:

            bad_files.append({

                "path":
                    path,

                "error":
                    f"{type(e).__name__}: {e}",

            })

        if (
            i + 1
        ) % 500 == 0:

            print(
                f"  checked "
                f"{i + 1}/"
                f"{len(scan_files)}..."
            )

    print()

    print(
        "Corrupt/unreadable files:"
    )

    print(
        f"{len(bad_files)} / "
        f"{len(scan_files)}"
    )

    for bad in bad_files[:20]:

        print(
            " ",
            bad
        )

    report["corruption_scan"] = {

        "mode":
            "full"
            if args.full
            else "sample",

        "checked":
            len(scan_files),

        "bad_count":
            len(bad_files),

        "bad_files":
            bad_files[:100],

    }

    # ========================================================
    # STEP 7
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7/7: Writing report")
    print("=" * 70)

    output_path = (
        "worldstrat_inspection_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str
        )

    print(
        f"Full report written to "
        f"{output_path}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"HR RGBN tiles: "
        f"{len(hr_rgbn_files)}"
    )

    print(
        f"LR L2A observations: "
        f"{len(lr_l2a_files)}"
    )

    print(
        f"Matched HR/LR observations: "
        f"{len(pairs)}"
    )

    print(
        f"Unreadable files in scan: "
        f"{len(bad_files)}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Use the REAL HR/LR compatibility results above "
        "to determine the correct SIRM training pipeline."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print(
            "\nScript crashed. Full traceback:"
        )

        traceback.print_exc()

        sys.exit(1)