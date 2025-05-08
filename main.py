from fastapi import FastAPI, UploadFile
from pathlib import Path
from PIL import Image
from predict import predict_class

app = FastAPI()


@app.post("/predict")
def model_inference(file: UploadFile):
    '''
    Returns the predicted class index when given an image
    '''
    # check if image and return error message if not
    if not "image" in file.content_type:
        return {"file_type": file.content_type, "error_message": "Not an image. Please upload image file."}

    # save the image locally
    img_file_path = Path(f"./{file.filename}")
    img = Image.open(file.file)
    img.save(img_file_path)
    img.close()

    # get the predicted class
    result = predict_class(img_file_path)

    # clean up and remove saved image
    Path.unlink(img_file_path)

    return {"predicted_class_index": str(result)}
