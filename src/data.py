import random

from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE


class WasteDataset(Dataset):
    def __init__(self, rows, train=False):
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10)]
        ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.rows = rows
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, label = self.rows[index]
        return self.transform(image.convert("RGB")), label


def split_rows(dataset, seed=42):
    rows = [(item["image"], int(item["label"])) for item in dataset]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]
