import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import onnx
import onnxruntime as ort
import load_model


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


def predict_class(image_path: Path):
    # get preprocessed input image
    input_img = preprocess_image(image_path)

    # get class labels
    with open(load_model.IMAGENET_CLASS_PATH, "r") as fp:
        class_labels = [label.strip() for label in fp.readlines()]

    # load model and run checker ## not sure if step is needed
    res18_model = onnx.load(load_model.RESNET18_MODEL_PATH)
    onnx.checker.check_model(res18_model)

    # start onnx runtime session
    ort_session = ort.InferenceSession(load_model.RESNET18_MODEL_PATH)

    output = ort_session.run(None, {'data': input_img.numpy()})
    output = np.squeeze(output[0])

    # log this
    print(class_labels[output.argmax(0)])
    return output.argmax(0)
