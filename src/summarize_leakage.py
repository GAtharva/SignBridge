
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "RealSign" / "Dataset"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

SPLITS = ["Training", "Validation", "Testing"]
hash_groups = defaultdict(list)

print("Analysing exact duplicate groups...")

for split in SPLITS:
    split_path = DATASET_ROOT / split

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                file_hash = hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest()

                hash_groups[file_hash].append({
                    "split": split,
                    "class": class_folder.name,
                    "path": str(image_path.relative_to(DATASET_ROOT))
                })

            except Exception as error:
                print(f"Error reading {image_path}: {error}")

cross_split_groups = []

for file_hash, images in hash_groups.items():
    splits_found = {item["split"] for item in images}

    if len(splits_found) > 1:
        cross_split_groups.append((file_hash, images))

pair_counts = Counter()
class_counts = Counter()
affected_images = Counter()
label_conflicts = []

for group_id, (file_hash, images) in enumerate(cross_split_groups, start=1):
    splits_found = sorted({item["split"] for item in images})
    classes_found = {item["class"] for item in images}

    pair_counts[" + ".join(splits_found)] += 1

    for class_name in classes_found:
        class_counts[class_name] += 1

    for split in splits_found:
        affected_images[split] += sum(
            item["split"] == split for item in images
        )

    if len(classes_found) > 1:
        label_conflicts.append((group_id, images))

# Export every file involved in cross-split duplicates.
csv_path = REPORT_DIR / "exact_duplicate_details.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "group_id", "sha256", "split", "class", "relative_path"
    ])

    for group_id, (file_hash, images) in enumerate(
        cross_split_groups, start=1
    ):
        for item in images:
            writer.writerow([
                group_id,
                file_hash,
                item["split"],
                item["class"],
                item["path"]
            ])

print("\n" + "=" * 60)
print("EXACT DATA LEAKAGE SUMMARY")
print("=" * 60)

print(f"\nCross-split duplicate groups: {len(cross_split_groups)}")

print("\nGroups by split combination:")
for combination, count in sorted(pair_counts.items()):
    print(f"{combination}: {count}")

print("\nGroups by class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nAffected image files by split:")
for split in SPLITS:
    print(f"{split}: {affected_images[split]}")

print(f"\nGroups with conflicting class labels: {len(label_conflicts)}")

print(f"\nDetailed report saved to:")
print(csv_path)

print("\nNo dataset images were modified or deleted.")
