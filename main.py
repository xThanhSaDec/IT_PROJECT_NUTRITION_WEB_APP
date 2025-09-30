from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # load image
    contents = await file.read()
    # TODO: gọi model.predict(contents)
    result = {
        "food_name": "Pho",
        "nutrition": {"calories": 350, "protein": 12, "fat": 5},
        "bbox": [50, 60, 200, 180]
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
