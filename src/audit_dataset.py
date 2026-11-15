from pathlib import Path
from collections import Counter
from PIL import Image
import imagehash
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "RealSign" / "Dataset"

SPLITS = ["Training", "Validation", "Testing"]

def audit_split(split_name):
    split_path = DATASET_ROOT / split_name

    print(f"\n{'=' * 55}")
    print(f"AUDITING: {split_name}")
    print(f"{'=' * 55}")

    class_counts = Counter()
    image_hashes = {}
    unreadable = []
    image_sizes = Counter()
    total = 0

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        image_files = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        class_counts[class_folder.name] = len(image_files)

        for image_path in tqdm(image_files, desc=f"Class {class_folder.name}"):
            total += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    image_sizes[img.size] += 1
                    image_hash = str(imagehash.phash(img.convert("RGB")))

                image_hashes.setdefault(image_hash, []).append(
                    str(image_path.relative_to(DATASET_ROOT))
                )

            except Exception as error:
                unreadable.append((str(image_path), str(error)))

    duplicate_groups = {
        image_hash: paths
        for image_hash, paths in image_hashes.items()
        if len(paths) > 1
    }

    print(f"\nTotal images: {total}")
    print(f"Number of classes: {len(class_counts)}")

    print("\nImages per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")

    print(f"\nUnique image dimensions: {len(image_sizes)}")
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Exact perceptual-hash duplicate groups: {len(duplicate_groups)}")

    if unreadable:
        print("\nUnreadable files:")
        for path, error in unreadable:
            print(path, error)

    if duplicate_groups:
        print("\nPotential duplicate images:")
        for paths in duplicate_groups.values():
            print(" - " + " | ".join(paths[:5]))

    return {
        "total": total,
        "classes": class_counts,
        "duplicates": duplicate_groups,
        "unreadable": unreadable,
    }


if __name__ == "__main__":
    results = {}

    for split in SPLITS:
        results[split] = audit_split(split)

    print("\n" + "=" * 55)
    print("DATASET AUDIT COMPLETED")
    print("=" * 55)