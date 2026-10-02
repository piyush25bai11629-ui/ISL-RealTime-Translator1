"""
ISL Real-Time Translator — minimal FastAPI backend.

One file, no ML dependencies. It implements the POST /predict contract the
frontend already speaks, so the "Live backend" mode on demo.html works today.

The classifier here is a MOTION HEURISTIC, not a trained model. It exists so the
wiring can be demonstrated before the CNN + LSTM weights land. Every response
carries source="heuristic" so nothing downstream can mistake it for a result.
Swap in the real model at load_model() / classify() and the contract is unchanged.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""

import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Contract — these must match js/demo.js and src/model.py
# --------------------------------------------------------------------------

FRAME_WINDOW = 30    # frames per sign sample, matches input_shape=(30, 258)
FEATURE_DIM = 258    # 132 pose + 63 left hand + 63 right hand
POSE_DIM = 132       # 33 pose landmarks x (x, y, z, visibility)

ACTIONS = ["hello", "thanks", "iloveyou"]

# Where the page is served from. The React frontend runs on Vite's 5173 in dev
# and 4173 under `npm run preview`. Add any other origin you serve it on.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app = FastAPI(
    title="ISL Real-Time Translator",
    version="0.1.0",
    description="Landmark window in, sign label out.",
)

# Without this the browser blocks the response and you see a CORS error
# instead of the prediction. The page and the API are different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------

class FrameData(BaseModel):
    """Body of POST /predict — a (30, 258) window of landmark floats."""
    landmarks: List[List[float]]


class Prediction(BaseModel):
    prediction: str
    status: str
    confidence: float
    inference_ms: float
    source: str


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_model():
    """
    Load the trained CNN + LSTM weights.

    TODO (Piyush, feature/piyush-ml): point this at the saved model, e.g.

        from tensorflow.keras.models import load_model as keras_load
        return keras_load("../models/isl_lstm.h5")

    Returning None keeps the server running on the heuristic below, so the
    frontend stays demonstrable while training is in progress.
    """
    return None


def classify(window: List[List[float]]):
    """
    Return (label, confidence).

    Real version, once load_model() returns a model:

        import numpy as np
        probs = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
        i = int(np.argmax(probs))
        return ACTIONS[i], float(probs[i])
    """
    if model is not None:
        raise NotImplementedError("Wire the trained model in here.")
    return heuristic(window)


def heuristic(window: List[List[float]]):
    """
    Mean hand displacement across the window, plus mean hand height in the last
    frame. Deliberately the same rule as mockPredict() in js/demo.js, so Mock
    and Live modes agree and any disagreement means a wiring bug, not a model
    difference.
    """
    hand_dims = FEATURE_DIM - POSE_DIM

    motion = 0.0
    for f in range(1, len(window)):
        prev, cur = window[f - 1], window[f]
        for i in range(POSE_DIM, FEATURE_DIM):
            motion += abs(cur[i] - prev[i])
    motion /= max(1, (len(window) - 1) * hand_dims)

    # Every third value from the hand blocks is a y coordinate.
    last = window[-1]
    ys = [last[i] for i in range(POSE_DIM + 1, FEATURE_DIM, 3) if last[i] != 0]
    height = sum(ys) / len(ys) if ys else 0.5

    if motion > 0.012:
        label = ACTIONS[0]        # hello — a wave, lots of movement
    elif height < 0.45:
        label = ACTIONS[2]        # iloveyou — hand held high
    else:
        label = ACTIONS[1]        # thanks

    confidence = min(0.99, 0.62 + min(motion * 18, 0.3))
    return label, confidence


model = load_model()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "ISL Real-Time Translator",
        "predict": "POST /predict",
        "expects": f"({FRAME_WINDOW}, {FEATURE_DIM}) landmark window",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    """Liveness probe. The frontend can hit this to show a backend indicator."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict_sign(data: FrameData):
    """
    Landmark window in, sign label out.

    Rejects the wrong shape loudly. A silent reshape here would produce
    confident nonsense, which is worse than a 400.
    """
    window = data.landmarks

    if len(window) != FRAME_WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {FRAME_WINDOW} frames, got {len(window)}.",
        )

    for n, frame in enumerate(window):
        if len(frame) != FEATURE_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {n} has {len(frame)} features, expected {FEATURE_DIM}.",
            )

    started = time.perf_counter()
    label, confidence = classify(window)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # inference_ms is the server half of research gap G6 (latency is asserted
    # far more often than it is measured). The page reports the client half.
    return Prediction(
        prediction=label,
        status="success",
        confidence=round(confidence, 4),
        inference_ms=round(elapsed_ms, 2),
        source="model" if model is not None else "heuristic",
    )
