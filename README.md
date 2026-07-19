# Print / Caption

A backend + frontend around your Flickr8k VGG16 + LSTM image captioning model.

- **backend/** — FastAPI server that loads your trained model and serves `/caption`
- **frontend/** — React (Vite) app: drop a photo in, watch the caption "develop"

## 1. Add your model files

Copy the two files you saved from the notebook into `backend/models/`:

```
backend/models/model.h5
backend/models/tokenizer.pkl
```

That's it — the server figures out `max_length` automatically from the model's
input shape, so you don't need to save anything else.

## 2. Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

First startup takes a bit longer — it downloads ImageNet weights for VGG16
(same as in your notebook) and loads your caption model. Once you see
`Ready. max_length=...` in the logs, check http://localhost:8000/health.

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Drop or click to select an image, then hit
**Develop caption**.

If you deploy the backend somewhere other than `localhost:8000`, copy
`.env.example` to `.env` in `frontend/` and set `VITE_API_URL` to your
backend's URL.

## API

`POST /caption` — multipart form field `file` (an image) → `{ "caption": "..." }`

`GET /health` — quick check that the model loaded, returns the detected `max_length`

## Notes

- The caption quality is only as good as the model you trained in the
  notebook (12 epochs on Flickr8k with BLEU-1/BLEU-2 as reported there) —
  this project doesn't change the model, just wraps it.
- CORS is wide open (`allow_origins=["*"]`) for local development. Lock this
  down in `backend/main.py` before deploying publicly.
