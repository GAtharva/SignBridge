
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "RealSign" / "Dataset"

SPLITS = ["Training", "Validation", "Testing"]

# Store image paths grouped by perceptual hash.
hash_groups = defaultdict(list)

print("Scanning all dataset splits...")

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

                hash_groups[phash].append({
                    "split": split,
                    "class": class_folder.name,
                    "path": str(image_path.relative_to(DATASET_ROOT))
                })

            except Exception as error:
                print(f"Error reading {image_path}: {error}")

# Keep groups that contain images from multiple splits.
cross_split_groups = []

for phash, images in hash_groups.items():
    splits_found = set(item["split"] for item in images)

    if len(splits_found) > 1:
        cross_split_groups.append((phash, images, splits_found))

# Count affected images and split combinations.
pair_counts = defaultdict(int)
affected_images = defaultdict(int)

for phash, images, splits_found in cross_split_groups:
    for split in splits_found:
        affected_images[split] += sum(
            1 for item in images if item["split"] == split
        )

    for pair in [
        ("Training", "Validation"),
        ("Training", "Testing"),
        ("Validation", "Testing")
    ]:
        if pair[0] in splits_found and pair[1] in splits_found:
            pair_counts[pair] += 1

print("\n" + "=" * 60)
print("CROSS-SPLIT DUPLICATE AUDIT")
print("=" * 60)

print(f"\nTotal cross-split hash groups: {len(cross_split_groups)}")

print("\nGroups shared between splits:")
for pair, count in pair_counts.items():
    print(f"{pair[0]} <-> {pair[1]}: {count}")

print("\nImages involved in cross-split groups:")
for split in SPLITS:
    print(f"{split}: {affected_images[split]}")

# Display a maximum of 20 examples.
print("\nSAMPLE CROSS-SPLIT GROUPS")
print("=" * 60)

for index, (phash, images, splits_found) in enumerate(
    cross_split_groups[:20], start=1
):
    print(f"\nGroup {index} | Splits: {', '.join(sorted(splits_found))}")

    for item in images:
        print(f"  {item['path']}")

print("\n" + "=" * 60)
print("AUDIT COMPLETED")
print("No dataset images were modified or deleted.")
