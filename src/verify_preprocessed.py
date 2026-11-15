
from pathlib import Path
from PIL import Image

DATASET_PATH = Path("datasets/processed/RealSign/Alphabet")
SPLITS = ["Training", "Validation", "Testing"]

expected_size = (224, 224)
total_images = 0
invalid_images = []

print("=" * 50)
print("SIGNBRIDGE - PREPROCESSED DATA VERIFICATION")
print("=" * 50)

for split in SPLITS:
    split_path = DATASET_PATH / split
    split_count = 0

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.glob("*.jpg"):
            split_count += 1

            try:
                with Image.open(image_path) as image:
                    if image.size != expected_size:
                        invalid_images.append(str(image_path))
            except Exception:
                invalid_images.append(str(image_path))

    total_images += split_count
    print(f"{split}: {split_count} images")

print("\nTotal images:", total_images)
print("Invalid images:", len(invalid_images))

if invalid_images:
    print("\nExamples of invalid files:")
    for path in invalid_images[:10]:
        print(path)
else:
    print("All images have the correct dimensions and can be opened!")

print("=" * 50)
