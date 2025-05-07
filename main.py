from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.post("/file_details")
async def get_file(file: UploadFile):
    content = await file.read()
    return {"filename": file.filename, "type": file.content_type, "file_size": len(content)}
