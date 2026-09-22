import random
import zipfile
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATASET_ARCHIVE, DATASET_NAME, IMAGE_SIZE


CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def prepare_dataset():
    cache_root = Path(".cache") / "trashnet"
    extract_root = cache_root / "dataset-resized"
    cache_root.mkdir(parents=True, exist_ok=True)

    if not extract_root.exists():
        archive = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=DATASET_ARCHIVE,
            repo_type="dataset",
        )
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(cache_root)

    candidates = [extract_root]
    candidates.extend(p for p in cache_root.rglob("*") if p.is_dir())

    for root in candidates:
        if all((root / name).is_dir() for name in CLASS_NAMES):
            rows = []
            for label, name in enumerate(CLASS_NAMES):
                for path in sorted((root / name).glob("*")):
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        rows.append((str(path), label))
            if rows:
                return rows, CLASS_NAMES

    raise RuntimeError("Could not locate the six TrashNet class folders after extraction.")


class WasteDataset(Dataset):
    def __init__(self, rows, train=False):
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10)]
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
    n = len(rows)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]
