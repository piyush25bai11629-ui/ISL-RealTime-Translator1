# ISL Real-Time Translator

Isolated Indian Sign Language recognition from a single RGB webcam. MediaPipe
extracts landmarks in the browser, a FastAPI service runs a trained CNN + LSTM
checkpoint over a 30-frame window, and the recognised sign is captioned and
spoken.

## Layout

```
backend/     FastAPI + TensorFlow. The only thing here that produces a prediction.
frontend/    React 19 + Vite + TypeScript + Tailwind. Capture, landmarks, caption, TTS.
upstream/    The original research checkout. Read-only reference.
```

Each half has its own README: [`backend/README.md`](backend/README.md) for the
artefact resolution order, endpoint contract and environment variables;
[`frontend/README.md`](frontend/README.md) for the component layout and tunables.

`upstream/` is a git checkout of
[piyush25bai11629-ui/ISL-RealTime-Translator1](https://github.com/piyush25bai11629-ui/ISL-RealTime-Translator1)
at `cc05522`, carrying the training code, the recorded `.npy` datasets and the
trained checkpoint. The backend reads its `best_isl_model.keras`,
`label_map.json` and datasets **in place** and never copies or modifies them, so
nothing in this repo can drift from the weights it claims to serve.

## There is no mock mode

An earlier build captioned signs with no model present. That was not a bug in
the model path — there were two prediction sources, and the UI defaulted to the
fake one:

- `frontend/src/lib/api.ts` exported `mockPredict()`, a local motion heuristic
  with no reject branch and a confidence floor of 0.62 — above the commit
  threshold, so trivial movement committed a sign and TTS spoke it.
- The backend's `load_model()` returned `None` and `classify()` silently fell
  through to a `heuristic()`, so even "live" mode was arithmetic.

Both are deleted, not disabled. `mockPredict`, the `'mock'` branch in
`useRecognition`, the `BackendMode` type, the Live/Mock toggle and the backend
`heuristic()` are all gone. If the server is down or the checkpoint failed to
load, the UI says so and predicts nothing — there is no substitute to fall back
to.

## Run it

Needs Python 3.10+ and Node 20+. Backend first; the frontend has nothing to talk
to otherwise.

```
cd backend
pip install -r requirements.txt
run.bat
```

Confirm the checkpoint actually loaded before opening the page:

```
curl http://localhost:8000/health
```

Expect `"model_loaded": true`. If it is `false`, read `load_error` — the
checkpoint was not found, or TensorFlow refused it. Then:

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access. The header shows two pills:
engine state, and backend state (`Model loaded` / `Backend up, model failed to
load` / `Backend unreachable on :8000`). Hover the second for the resolved model
path.

Point the frontend elsewhere with `VITE_API_BASE`:

```
set VITE_API_BASE=http://192.168.1.20:8000
npm run dev
```

The backend's CORS allow-list covers Vite dev (5173) and preview (4173). Serving
the page from any other origin means adding it to `ALLOWED_ORIGINS` in
`backend/app.py`, or the browser blocks the response before your code sees it.

## Vocabulary

Five classes — the only ones the checkpoint was trained on:

| Slug | Display |
| --- | --- |
| `i_love_you` | I Love You |
| `yes` | Yes |
| `no` | No |
| `namaste` | Namaste |
| `thank_you` | Thank You |

Slugs go on the wire, display strings go to humans. The model file declares 50
output units; only ids 45–49 were trained, so the backend masks the rest and
renormalises the softmax over those five.

## Accuracy, stated plainly

Upstream training reports **86.7% validation accuracy** — but on a same-session
holdout of 150 sequences (5 classes × 30 takes) recorded by one signer, in one
room, in one sitting. Replaying that same `.npy` corpus through this service
gives:

| Class | Correct | Mean confidence |
| --- | --- | --- |
| Namaste | 30/30 | 0.990 |
| Thank You | 30/30 | 0.966 |
| No | 29/30 | 0.602 |
| I Love You | 17/30 | 0.436 |
| Yes | 16/30 | 0.427 |
| **Total** | **122/150 (81.3%)** | |

Reproduce it with `python infer_cli.py --dataset` in `backend/`.

*I Love You* and *Yes* are genuinely confused by the checkpoint and usually fall
below the 0.6 commit threshold, so they route to the correction bar rather than
committing. That is the trained weights, not the wiring — upstream's own
confusion matrix shows the same pair. Retraining is the fix; nothing in this
codebase can paper over it honestly.

Live accuracy with a different signer, camera and lighting will be materially
lower than either figure. Signer-independent evaluation needs recordings from
signers who are not in the training set, which this dataset does not contain.

## Two ways in

**Live webcam.** `useRecognition` runs on `requestAnimationFrame`: extract
landmarks → push a 258-float vector into a 30-frame ring buffer → once full,
POST to `/predict` → require `STABILITY_FRAMES` agreeing predictions above
`CONFIDENCE_THRESHOLD` → commit on a pause.

Requests are throttled to one per `PREDICT_INTERVAL_MS` (300 ms). A forward pass
costs ~104 ms median / 145 ms p95 on CPU and the backend serialises them behind a
lock, so one request per captured frame would enqueue faster than they drain and
the caption would fall further behind the camera every second.

**Recorded clip.** The *Recognise a recorded clip* panel uploads a short video to
`/predict/video`. The server decodes it, runs MediaPipe Holistic over the frames,
rebuilds the same vectors used in training, resamples to 30 frames and calls the
same classifier — so a clip and a live window of the same sign score identically.
This is the practical way to check the model against a known recording.

## Feature dimensions: 258 vs 225

The browser emits **258** floats per frame (pose 33×4 *with* visibility, then
left and right hand 21×3). The model was trained on **225** (pose 33×3, no
visibility). The backend accepts either width and strips the visibility channel;
both paths are verified to produce bit-identical model input. Sending the
browser's native layout keeps that reshape in exactly one place.

## Not predicting when nobody is signing

If fewer than `ISL_MIN_HAND_FRAMES` (default 10) of the 30 frames contain a hand,
the backend returns `status: "idle"` with an empty prediction and a reason,
instead of an argmax over noise. An all-zero window returns
`"hands visible in 0/30 frames, need 10"`. The client clears the caption on
`idle` and resets the stability streak.

## Corrections

A prediction that is stable but below threshold is offered for correction rather
than committed. Corrections are keyed by slug — relabelling the UI cannot
corrupt the log — and buffered in `localStorage` under `isl.corrections.v1`.
There is no ingest endpoint yet; *Export corrections* downloads them as JSON and
clears the buffer.

## Scope

Isolated sign recognition over a five-sign vocabulary. Output is a sequence of
recognised signs, not a grammatical translation — sentence-level ISL corpora
remain scarce. The known limits, in order of how much they would change the
numbers above: single-signer training data, a five-class vocabulary, no
non-manual markers (face landmarks are excluded from the feature vector), and no
signer-independent test split.
