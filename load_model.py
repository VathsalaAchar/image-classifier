from pathlib import Path
from utils import download_file

RESNET18_URL = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx?download="
RESNET18_MODEL_PATH = Path("./resnet18.onnx")

if not RESNET18_MODEL_PATH.is_file():
    download_file(RESNET18_URL, RESNET18_MODEL_PATH)


IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_CLASS_PATH = Path("./imagenet_classes.txt")

if not IMAGENET_CLASS_PATH.is_file():
    download_file(IMAGENET_CLASSES_URL, IMAGENET_CLASS_PATH)
