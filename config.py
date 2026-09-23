DATASET_NAME = "garythung/trashnet"
DATASET_ARCHIVE = "dataset-resized.zip"

# Expanded 14-class waste taxonomy
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "food_vegetable_waste",
    "fruit_waste",
    "leaves_organic",
    "clothes",
    "batteries",
    "electronics",
    "wood",
    "chemical_waste",
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
SEED = 42
MAX_IMAGES_PER_SOURCE_CLASS = 500

MODEL_PATH = "artifacts/waste_mobilenetv3.pth"
CLASS_NAMES_PATH = "artifacts/class_names.txt"
