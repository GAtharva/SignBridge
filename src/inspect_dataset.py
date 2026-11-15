
from pathlib import Path

# Update this path if your folder structure differs
DATASET_PATH = Path("datasets/raw/RealSign/Dataset")

SPLITS = ["Training", "Validation", "Testing"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 55)
print("         SIGNBRIDGE - DATASET INSPECTION")
print("=" * 55)

total_images = 0

for split in SPLITS:
    split_path = DATASET_PATH / split

    print(f"\n{split.upper()}")

    if not split_path.exists():
        print(f"Folder not found: {split_path}")
        continue

    split_total = 0

    for class_folder in sorted(split_path.iterdir()):
        if class_folder.is_dir():
            images = [
                file for file in class_folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            count = len(images)
            split_total += count

            print(f"{class_folder.name}: {count} images")

    print(f"Total {split}: {split_total}")

    total_images += split_total

print("\n" + "=" * 55)
print("TOTAL IMAGES:", total_images)
print("=" * 55)
