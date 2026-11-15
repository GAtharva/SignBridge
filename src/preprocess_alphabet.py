
from pathlib import Path
from PIL import Image, ImageOps

# Project paths
SOURCE_PATH = Path("datasets/raw/RealSign/Dataset")
OUTPUT_PATH = Path("datasets/processed/RealSign/Alphabet")

# Configuration
IMAGE_SIZE = 224
SPLITS = ["Training", "Validation", "Testing"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(image_path, output_path):
    try:
        with Image.open(image_path) as image:
            # Convert image to RGB
            image = image.convert("RGB")

            # Resize while preserving aspect ratio and add padding
            image = ImageOps.pad(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
                centering=(0.5, 0.5)
            )

            # Save processed image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="JPEG", quality=95)

            return True

    except Exception as error:
        print(f"Error processing {image_path}: {error}")
        return False


def main():
    total_processed = 0
    total_failed = 0

    print("=" * 55)
    print("       SIGNBRIDGE - ALPHABET PREPROCESSING")
    print("=" * 55)

    for split in SPLITS:
        split_path = SOURCE_PATH / split
        print(f"\nProcessing {split}...")

        if not split_path.exists():
            print(f"Missing folder: {split_path}")
            continue

        split_processed = 0

        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir():
                continue

            for image_path in class_folder.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                output_file = (
                    OUTPUT_PATH
                    / split
                    / class_folder.name
                    / f"{image_path.stem}.jpg"
                )

                success = preprocess_image(image_path, output_file)

                if success:
                    split_processed += 1
                else:
                    total_failed += 1

        total_processed += split_processed
        print(f"Processed {split}: {split_processed} images")

    print("\n" + "=" * 55)
    print("PREPROCESSING SUMMARY")
    print("=" * 55)
    print("Total processed:", total_processed)
    print("Total failed:", total_failed)
    print("Image size:", f"{IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Output folder:", OUTPUT_PATH)
    print("=" * 55)


if __name__ == "__main__":
    main()
