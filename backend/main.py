import io
import os
import pickle

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CAPTION_MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")

app = FastAPI(title="Image Caption Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {}


@app.on_event("startup")
def load_artifacts():
    if not os.path.exists(CAPTION_MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise RuntimeError(
            f"Missing model files. Place model.h5 and tokenizer.pkl inside {MODELS_DIR}"
        )

    print("Loading VGG16 feature extractor...")
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    state["vgg"] = vgg

    print("Loading caption model...")
    caption_model = load_model(CAPTION_MODEL_PATH)
    state["caption_model"] = caption_model

    # Some Keras versions expose the text input as an InputLayer without a usable
    # .input_shape attribute, so infer max_length from the text input tensor directly.
    max_length = None
    for tensor in caption_model.inputs:
        name = getattr(tensor, "name", "")
        if name.split(":")[0] == "text":
            shape = tuple(tensor.shape)
            if len(shape) >= 2 and shape[1] is not None:
                max_length = int(shape[1])
                break

    if max_length is None:
        # Fall back to any non-image text-like input shape.
        for tensor in caption_model.inputs:
            shape = tuple(tensor.shape)
            if len(shape) >= 2 and shape[1] is not None and shape[1] != 4096:
                max_length = int(shape[1])
                break

    if max_length is None:
        raise RuntimeError("Could not determine the caption sequence length from the loaded model.")

    state["max_length"] = max_length

    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        state["tokenizer"] = pickle.load(f)

    print(f"Ready. max_length={state['max_length']}")


def index_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


def extract_features(image: Image.Image, vgg_model) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = img_to_array(image)
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    arr = preprocess_input(arr)
    return vgg_model.predict(arr, verbose=0)


def predict_caption(feature, caption_model, tokenizer, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = caption_model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = index_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()


@app.get("/health")
def health():
    return {"status": "ok", "max_length": state.get("max_length")}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    feature = extract_features(image, state["vgg"])
    caption = predict_caption(
        feature, state["caption_model"], state["tokenizer"], state["max_length"]
    )

    return JSONResponse({"caption": caption})
