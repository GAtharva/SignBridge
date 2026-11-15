
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "RealSign" / "Dataset"

SPLITS = ["Training", "Validation", "Testing"]
hash_groups = defaultdict(list)

print("Checking exact file duplicates using SHA-256...")

for split in SPLITS:
    split_path = DATASET_ROOT / split

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

                hash_groups[sha256].append({
                    "split": split,
                    "class": class_folder.name,
                    "path": str(image_path.relative_to(DATASET_ROOT))
                })

            except Exception as error:
                print(f"Error: {image_path} - {error}")

cross_split_duplicates = []

for sha256, images in hash_groups.items():
    splits_found = {item["split"] for item in images}

    if len(splits_found) > 1:
        cross_split_duplicates.append(images)

print("\n" + "=" * 60)
print("EXACT CROSS-SPLIT DUPLICATE REPORT")
print("=" * 60)

print(f"\nExact cross-split duplicate groups: {len(cross_split_duplicates)}")

for index, images in enumerate(cross_split_duplicates[:30], start=1):
    print(f"\nGroup {index}")

    for item in images:
        print(f"  {item['path']}")

print("\n" + "=" * 60)
print("AUDIT COMPLETED")
print("No images were modified or deleted.")
