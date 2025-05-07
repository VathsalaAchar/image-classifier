import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image


def preprocess_image(image_path: Path):
    preprocess = v2.Compose([
        v2.Resize(size=256),
        v2.CenterCrop(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    # run the image through the transforms to preprocess it
    img_tensor = preprocess(img)
    # add a dimension to create minibatch as expected by model
    img_batch = img_tensor.unsqueeze(0)

    return img_batch
