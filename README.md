# ONNX Model Inference with FastAPI

This is a small FastAPI application that has a `/predict` endpoint for classifiying images using the Resnet-18 ONNX model.

The model and the imagenet classes are downloaded when the server is started. 

### Run the Webserver locally

```bash
fastapi dev main.py
```


### Test the API


```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path/to/image.jpg"
```

#### Result:

```json
{
  "predicted_class_index": 123
}
```

