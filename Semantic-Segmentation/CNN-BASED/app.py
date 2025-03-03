import io
import base64
import uvicorn
import numpy as np
from PIL import Image
from inference import load_model, predict
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse

app = FastAPI()
session = load_model()

def encode_image(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    encoded = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

@app.post("/predict", response_class=HTMLResponse)
async def predict_image(
    file: UploadFile = File(...), 
    postprocess: bool = Query(default=False, description="Enable or disable post-processing")
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    mask, overlay = predict(image, postprocess)

    mask_pil = Image.fromarray(mask.astype(np.uint8))
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))

    mask_encoded = encode_image(mask_pil)
    overlay_encoded = encode_image(overlay_pil)

    html_content = f"""
    <html>
    <body>
        <h2>Segmentation Results</h2>
        <h3>Mask:</h3>
        <img src="{mask_encoded}" alt="Mask Image"/>
        <h3>Overlay:</h3>
        <img src="{overlay_encoded}" alt="Overlay Image"/>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
