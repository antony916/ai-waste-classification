import random
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    CLASS_NAMES,
    DATASET_ARCHIVE,
    DATASET_NAME,
    IMAGE_SIZE,
    MAX_IMAGES_PER_SOURCE_CLASS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Public sources used to expand the taxonomy:
# - TrashNet: original six recyclable/residual classes.
# - cpoisson/trash-optimizer-dataset: real-world cardboard, food organics,
#   glass, metal, miscellaneous trash, paper, plastic, textile, vegetation,
#   battery and wood.
# - huaweilin/waste-classification: cleaned photos containing batteries,
#   e-waste, paints/pesticides, food scraps, kitchen waste and yard trimmings.
# - waste_pictures (Kaggle): genuine fruit-waste imagery such as watermelon rind.
#
# The fruit-waste source is optional at code level but required before training:
# the model must have actual fruit-waste examples rather than fresh-fruit photos.

CPOISSON_DATASET = "cpoisson/trash-optimizer-dataset"
HUAWEILIN_DATASET = "huaweilin/waste-classification"
FRUIT_KAGGLE_DATASET = "wangziang/waste-pictures"


def _safe_name(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _save_image(image, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.save(destination, format="JPEG", quality=92)


def _add_file(rows, counts, source, target, path, limit):
    if target not in CLASS_NAMES or not path.exists():
        return
    if counts[(source, target)] >= limit:
        return
    rows.append((str(path), CLASS_NAMES.index(target)))
    counts[(source, target)] += 1


def _prepare_trashnet(cache_root, rows, counts):
    source = "trashnet"
    extract_root = cache_root / "trashnet" / "dataset-resized"
    extract_root.parent.mkdir(parents=True, exist_ok=True)

    if not extract_root.exists():
        archive = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=DATASET_ARCHIVE,
            repo_type="dataset",
        )
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_root.parent)

    for target in CLASS_NAMES[:6]:
        folder = extract_root / target
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                _add_file(rows, counts, source, target, path, MAX_IMAGES_PER_SOURCE_CLASS)


def _prepare_cpoisson(cache_root, rows, counts):
    source = "cpoisson"
    dataset = load_dataset(
        CPOISSON_DATASET,
        data_dir="dataset",
        split="train",
    )
    label_names = dataset.features["label"].names

    mapping = {
        "cardboard": "cardboard",
        "food_organics": "food_vegetable_waste",
        "glass": "glass",
        "metal": "metal",
        "miscellaneous_trash": "trash",
        "paper": "paper",
        "plastic": "plastic",
        "textile_trash": "clothes",
        "vegetation": "leaves_organic",
        "battery": "batteries",
        "car_battery": "batteries",
        "wood": "wood",
    }

    for index, item in enumerate(dataset):
        source_label = _safe_name(label_names[item["label"]])
        target = mapping.get(source_label)
        if not target or counts[(source, target)] >= MAX_IMAGES_PER_SOURCE_CLASS:
            continue
        image = item["image"]
        destination = (
            cache_root
            / source
            / target
            / f"{source}_{target}_{index:06d}.jpg"
        )
        _save_image(image, destination)
        _add_file(rows, counts, source, target, destination, MAX_IMAGES_PER_SOURCE_CLASS)


def _prepare_huaweilin(cache_root, rows, counts):
    source = "huaweilin"
    dataset = load_dataset(HUAWEILIN_DATASET, split="cleaned")

    mapping = {
        "batteries": "batteries",
        "e_waste": "electronics",
        "paints": "chemical_waste",
        "pesticides": "chemical_waste",
        "food_scraps": "food_vegetable_waste",
        "kitchen_waste": "food_vegetable_waste",
        "yard_trimmings": "leaves_organic",
        "cans_all_type": "metal",
        "glass_containers": "glass",
        "paper_products": "paper",
        "plastic_bottles": "plastic",
    }

    for index, item in enumerate(dataset):
        source_label = _safe_name(item["subclass"])
        target = mapping.get(source_label)
        if not target or counts[(source, target)] >= MAX_IMAGES_PER_SOURCE_CLASS:
            continue
        image = item["image"]
        destination = (
            cache_root
            / source
            / target
            / f"{source}_{target}_{index:06d}.jpg"
        )
        _save_image(image, destination)
        _add_file(rows, counts, source, target, destination, MAX_IMAGES_PER_SOURCE_CLASS)


def _prepare_fruit_waste(cache_root, rows, counts):
    source = "waste_pictures"
    target = "fruit_waste"
    existing = list((cache_root / source / target).glob("*.jpg"))
    for path in existing[:MAX_IMAGES_PER_SOURCE_CLASS]:
        _add_file(rows, counts, source, target, path, MAX_IMAGES_PER_SOURCE_CLASS)

    if counts[(source, target)] >= 30:
        return

    try:
        import kagglehub
        dataset_root = Path(kagglehub.dataset_download(FRUIT_KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            "Fruit-waste images are required. The waste_pictures Kaggle dataset "
            "contains a dedicated watermelon-rind waste class. Install kagglehub "
            "and configure Kaggle access, or place the downloaded waste_pictures "
            "folder under .cache/waste14/waste_pictures/."
        ) from exc

    candidates = []
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            normalized = _safe_name(path.parent.name)
            if "watermelon" in normalized and "rind" in normalized:
                candidates.append(path)

    if not candidates:
        raise RuntimeError(
            "Could not find the watermelon-rind class inside waste_pictures. "
            "Place genuine fruit-waste images under .cache/waste14/"
            "waste_pictures/fruit_waste/ and rerun training."
        )

    for index, path in enumerate(candidates):
        if counts[(source, target)] >= MAX_IMAGES_PER_SOURCE_CLASS:
            break
        destination = (
            cache_root
            / source
            / target
            / f"{source}_{target}_{index:06d}.jpg"
        )
        if not destination.exists():
            with Image.open(path) as image:
                _save_image(image, destination)
        _add_file(rows, counts, source, target, destination, MAX_IMAGES_PER_SOURCE_CLASS)

    if counts[(source, target)] < 30:
        raise RuntimeError(
            f"Only {counts[(source, target)]} fruit-waste images were found. "
            "At least 30 are required for a meaningful 14-class training run."
        )


def _build_manifest(cache_root):
    rows = []
    counts = defaultdict(int)

    _prepare_trashnet(cache_root, rows, counts)
    _prepare_cpoisson(cache_root, rows, counts)
    _prepare_huaweilin(cache_root, rows, counts)
    _prepare_fruit_waste(cache_root, rows, counts)

    per_class = defaultdict(int)
    for _, label in rows:
        per_class[CLASS_NAMES[label]] += 1

    missing = [name for name in CLASS_NAMES if per_class[name] == 0]
    if missing:
        raise RuntimeError(f"No images were prepared for classes: {missing}")

    manifest = cache_root / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as file:
        for path, label in rows:
            file.write(f"{label}\t{path}\n")

    print("Prepared 14-class waste dataset:")
    for name in CLASS_NAMES:
        print(f"  {name}: {per_class[name]} images")

    return rows, CLASS_NAMES


def prepare_dataset():
    cache_root = Path(".cache") / "waste14"
    manifest = cache_root / "manifest.tsv"
    cache_root.mkdir(parents=True, exist_ok=True)

    if manifest.exists():
        rows = []
        with manifest.open("r", encoding="utf-8") as file:
            for line in file:
                label, path = line.rstrip("\n").split("\t", 1)
                if Path(path).exists():
                    rows.append((path, int(label)))
        if rows and all(any(label == CLASS_NAMES.index(name) for _, label in rows) for name in CLASS_NAMES):
            return rows, CLASS_NAMES

    return _build_manifest(cache_root)


class WasteDataset(Dataset):
    def __init__(self, rows, train=False):
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.12,
                ),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.rows = rows
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, label = self.rows[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def split_rows(rows, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)

    # Stratified 80/10/10 split so every class is represented in each split.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    for class_rows in grouped.values():
        random.Random(seed).shuffle(class_rows)
        n = len(class_rows)
        train_end = max(1, int(0.80 * n))
        val_end = max(train_end + 1, int(0.90 * n))
        val_end = min(val_end, n - 1) if n >= 3 else n

        train_rows.extend(class_rows[:train_end])
        val_rows.extend(class_rows[train_end:val_end])
        test_rows.extend(class_rows[val_end:])

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(val_rows)
    random.Random(seed + 2).shuffle(test_rows)
    return train_rows, val_rows, test_rows
