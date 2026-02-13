"""
Download training data from Roboflow and fix filenames for SAM2 compatibility.

Reads credentials from environment variables:
  ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION

Usage:
  python scripts/download_data.py [--data-dir /path/to/data/train]
"""

import os
import re
import sys
import shutil
import argparse


def download_dataset(api_key: str, workspace: str, project: str, version: int, dest_dir: str):
    """Download dataset from Roboflow in SAM2 format."""
    from roboflow import Roboflow

    print(f"[1/3] Connecting to Roboflow workspace '{workspace}', project '{project}' v{version}...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    print(f"[2/3] Downloading dataset in 'sam2' format...")
    dataset = ver.download("sam2")

    # Move downloaded files into dest_dir
    src = dataset.location
    print(f"[2/3] Moving data from '{src}' → '{dest_dir}'...")

    os.makedirs(dest_dir, exist_ok=True)

    # If Roboflow exports into a subfolder (e.g. train/), flatten it
    train_sub = os.path.join(src, "train")
    actual_src = train_sub if os.path.isdir(train_sub) else src

    for f in os.listdir(actual_src):
        shutil.move(os.path.join(actual_src, f), os.path.join(dest_dir, f))

    # Cleanup empty download folder
    shutil.rmtree(src, ignore_errors=True)
    print(f"[2/3] Download complete. {len(os.listdir(dest_dir))} files in '{dest_dir}'")


def fix_filenames(folder: str):
    """
    Two-step filename fix for SAM2 compatibility:

    Step 1 — Strip '.rf.<hash>' from Roboflow exports:
      floor_plan_image_1000_jpg.rf.a1b2c3d4e5f6.jpg  →  floor_plan_image_1000_jpg.jpg

    Step 2 — Reformat dots to underscores & ensure '_<number>.<ext>' suffix:
      SAM2 expects filenames like 'name_1.jpg'. Extra dots confuse the loader.
      image.v2.1000.jpg  →  image_v2_1000_1.jpg
    """
    # --- Step 1: Strip .rf.<hash> ---
    step1_count = 0
    for f in os.listdir(folder):
        if ".rf." in f:
            new_name = re.sub(r"\.rf\.[a-f0-9]+", "", f)
            os.rename(os.path.join(folder, f), os.path.join(folder, new_name))
            step1_count += 1

    if step1_count:
        print(f"[3/4] Stripped .rf.<hash> from {step1_count} files")

    # --- Step 2: Reformat dots → underscores, ensure _<number>.<ext> ---
    step2_count = 0
    for filename in os.listdir(folder):
        # Replace all dots except the last one (before extension) with underscores
        new_filename = filename.replace(".", "_", filename.count(".") - 1)
        # Ensure filename ends with _<number>.<ext> (SAM2 requirement)
        if not re.search(r"_\d+\.\w+$", new_filename):
            new_filename = new_filename.replace(".", "_1.")
        if new_filename != filename:
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
            step2_count += 1

    if step2_count:
        print(f"[4/4] Reformatted {step2_count} filenames (dots → underscores, added _<num> suffix)")

    # Show sample
    sample = sorted(os.listdir(folder))[:6]
    print("      Sample filenames:")
    for s in sample:
        print(f"        {s}")


def main():
    parser = argparse.ArgumentParser(description="Download & prepare SAM2 training data")
    parser.add_argument(
        "--data-dir",
        default="/app/fine_tuning_SAM2/data/train",
        help="Destination directory for training data (default: container path)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    # Check if data already exists
    existing = [f for f in os.listdir(data_dir) if f.endswith((".jpg", ".json"))] if os.path.isdir(data_dir) else []
    if len(existing) > 10:
        print(f"✅ Data already present ({len(existing)} files in '{data_dir}'). Skipping download.")
        # Still fix filenames in case they haven't been fixed
        if any(".rf." in f for f in existing):
            fix_filenames(data_dir)
        return

    # Read credentials from environment
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")
    project = os.environ.get("ROBOFLOW_PROJECT")
    version = int(os.environ.get("ROBOFLOW_VERSION", "1"))

    if not all([api_key, workspace, project]):
        print("❌ Missing environment variables. Set these in .env:")
        print("   ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION")
        sys.exit(1)

    print("=" * 60)
    print("  SAM2 Training Data — Download & Prepare")
    print("=" * 60)

    download_dataset(api_key, workspace, project, version, data_dir)
    fix_filenames(data_dir)

    total_jpg = len([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    total_json = len([f for f in os.listdir(data_dir) if f.endswith(".json")])
    print(f"\n✅ Done! {total_jpg} images, {total_json} annotations in '{data_dir}'")


if __name__ == "__main__":
    main()
