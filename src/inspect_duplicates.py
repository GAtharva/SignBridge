
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "RealSign" / "Dataset"

SPLITS = ["Training", "Validation", "Testing"]

# Collect identical perceptual hashes across ALL splits.
hash_groups = defaultdict(list)

for split in SPLITS:
    split_path = DATASET_ROOT / split

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                with Image.open(image_path) as img:
                    phash = str(imagehash.phash(img.convert("RGB")))

                hash_groups[phash].append(
                    str(image_path.relative_to(DATASET_ROOT))
                )

            except Exception as error:
                print(f"Could not read {image_path}: {error}")

# Keep groups containing images from at least two files.
duplicate_groups = {
    phash: paths
    for phash, paths in hash_groups.items()
    if len(paths) > 1
}

# Sort groups by number of images, largest first.
sorted_groups = sorted(
    duplicate_groups.items(),
    key=lambda item: len(item[1]),
    reverse=True
)

print("\nPOTENTIAL DUPLICATE GROUPS")
print("=" * 60)

for index, (phash, paths) in enumerate(sorted_groups[:30], start=1):
    print(f"\nGroup {index} | Images: {len(paths)}")
    for path in paths:
        print(f"  {path}")

print("\n" + "=" * 60)
print(f"Total potential duplicate groups: {len(duplicate_groups)}")
print("No images were modified or deleted.")
