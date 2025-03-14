from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import keras
import os

app = FastAPI()

model_path = os.path.join("models", "AI_TomatGrader.keras")
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
else:
    raise IOError("Unable to load model")

class_names = ["Reject", "Ripe", "Unripe"]


class ImageData(BaseModel):
    image_base64: str


class Classification(BaseModel):
    predicted_class: str


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AI TomatGrader REST API",
        version="1.0.0",
        summary="This API is designed for ESP32 to send tomato images for tomato classification.",
        description="I am **NOOB**. Please don't attack me.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.post("/analyze")
async def analyze_image(data: ImageData) -> Classification:
    try:
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(BytesIO(image_data))
        image = image.resize((256, 256))
        image_array = np.asarray(image)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        return Classification(predicted_class=predicted_class)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
