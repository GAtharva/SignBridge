from pathlib import Path
from PIL import Image

DATASET_PATH = Path("datasets/raw/RealSign/Dataset")

SPLITS = ["Training", "Validation", "Testing"]

corrupted_images = []
total_checked = 0
image_sizes = {}

for split in SPLITS:
    split_path = DATASET_PATH / split

    print(f"\nChecking {split}...")

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            total_checked += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    image_sizes[img.size] = image_sizes.get(img.size, 0) + 1

            except Exception:
                corrupted_images.append(str(image_path))

print("\n" + "=" * 50)
print("IMAGE QUALITY REPORT")
print("=" * 50)

print("Total images checked:", total_checked)
print("Corrupted images:", len(corrupted_images))

print("\nImage dimension summary:")
print("Unique dimensions:", len(image_sizes))

print("\nTop 10 most common dimensions:")
for size, count in sorted(
    image_sizes.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:
    print(f"{size}: {count} images")

if corrupted_images:
    print("\nCorrupted image paths:")
    for path in corrupted_images:
        print(path)
else:
    print("\nAll images opened successfully!")

print("=" * 50)