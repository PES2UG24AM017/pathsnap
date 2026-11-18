"""
api.py  -  PathSnap FastAPI Backend (Laptop 1)
===============================================
Endpoints:
  GET  /health             - liveness + model status
  GET  /trips              - list evaluation trips
  POST /predict/{trip_id}  - run map-matching for one trip
  POST /predict            - same, body {"trip_id": "..."}   (used by index.html)

Prediction = LightGBM candidate scores -> Viterbi (classifier+HMM).
Falls back to the plain HMM if the model file is missing or fails to load.

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000
Evaluator data location (default ../evaluator_test/evaluator_test):
  PATHSNAP_EVAL_DIR=/path/to/evaluator_test  uvicorn api:app ...
"""

import os, sys, time, threading
import joblib
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import load_nodes, load_track
from pipeline import run_trip

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(BASE_DIR)
EVAL_DIR  = os.environ.get('PATHSNAP_EVAL_DIR',
                           os.path.join(WORKSPACE, 'evaluator_test', 'evaluator_test'))
MODEL_PATH = os.environ.get('PATHSNAP_MODEL',
                            os.path.join(BASE_DIR, 'models', 'best_model.joblib'))

# ── model ─────────────────────────────────────────────────────────────────────
_clf = None
try:
    _clf = joblib.load(MODEL_PATH)
    print(f"[API] Loaded model: {type(_clf).__name__} from {MODEL_PATH}")
except Exception as e:                      # missing file, missing lightgbm, ...
    print(f"[API] Could not load model ({e}) - running HMM-only")

# The Dijkstra cache in map_matching is a module-level dict, so two requests
# must never run at the same time.
_lock = threading.Lock()

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PathSnap API",
              description="Map-matching and route reconstruction for GPS trajectories",
              version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


class TripInfo(BaseModel):
    trip_id:        str
    gps_points:     int
    n_observations: int
    nodes:          int
    arcs:           int


# ── helpers ───────────────────────────────────────────────────────────────────

def list_trips():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR)
                  if os.path.isdir(os.path.join(EVAL_DIR, d)))


def _segments(route, nodes, arcs, gt_set):
    """Arc ids -> [{arc_id, coords:[[lat,lon],[lat,lon]], correct}] for Leaflet."""
    segs = []
    for a in route:
        s, d = arcs[a]
        segs.append({
            'arc_id':  a,
            'coords':  [[nodes[s][1], nodes[s][0]], [nodes[d][1], nodes[d][0]]],
            'correct': (a in gt_set) if gt_set else None,
        })
    return segs


def _clean(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _predict(trip_id: str):
    if trip_id not in list_trips():          # also blocks path traversal
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found")
    folder = os.path.join(EVAL_DIR, trip_id)

    try:
        with _lock:
            res = run_trip(folder, trip_id, _clf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    nodes, arcs, track = res['nodes'], res['arcs'], res['track']
    gt_path = os.path.join(folder, trip_id + '.route')
    gt_set = set()
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt_set = {int(l) for l in f if l.strip()}

    # "final" = classifier+HMM when the model is loaded, else HMM-only
    final_key = 'classifier_hmm' if 'classifier_hmm' in res['routes'] else 'hmm_only'
    final     = res['routes'][final_key]
    fm        = res['metrics'][final_key]
    t_final   = res['timing'].get('classifier_hmm_s', res['timing']['hmm_only_s'])

    routes = {'hmm': {'segments': _segments(final, nodes, arcs, gt_set)}}
    metrics = {'hmm': _clean(fm)}
    if 'classifier_only' in res['routes']:
        routes['classifier_only'] = {'segments': _segments(
            res['routes']['classifier_only'], nodes, arcs, gt_set)}
        metrics['classifier_only'] = _clean(res['metrics']['classifier_only'])
        metrics['hmm_only'] = _clean(res['metrics']['hmm_only'])
        if res.get('candidate_miss_rate') is not None:
            metrics['hmm']['candidate_miss_rate'] = round(res['candidate_miss_rate'], 4)

    t0 = track[0][2] if track else 0.0
    return {
        # --- used by index.html ---
        'trip_id':        trip_id,
        'n_observations': len(track),
        'inference_ms':   round(t_final * 1000, 1),
        'gps':            [[lat, lon, round(ts - t0, 1)] for lon, lat, ts in track],
        'routes':         routes,
        'metrics':        metrics,
        # --- flat summary (original response shape) ---
        'predicted_route':  final,
        'num_arcs':         len(final),
        'precision':        fm.get('precision'),
        'recall':           fm.get('recall'),
        'f1_score':         fm.get('macro_f1'),
        'inference_time_s': round(t_final, 3),
        'mode':             'classifier+hmm' if final_key == 'classifier_hmm' else 'hmm_only',
    }


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": _clf is not None,
            "model_type": type(_clf).__name__ if _clf else "none",
            "eval_dir": EVAL_DIR,
            "eval_trips": len(list_trips())}


@app.get("/trips", response_model=List[TripInfo])
def get_trips():
    out = []
    for tid in list_trips():
        base = os.path.join(EVAL_DIR, tid, tid)
        try:
            n_nodes = len(load_nodes(base + '.nodes'))
            n_gps   = len(load_track(base + '.track'))
            with open(base + '.arcs') as f:
                n_arcs = sum(1 for l in f if l.strip())
        except Exception:
            n_nodes = n_gps = n_arcs = 0
        out.append(TripInfo(trip_id=tid, gps_points=n_gps, n_observations=n_gps,
                            nodes=n_nodes, arcs=n_arcs))
    return out


@app.post("/predict/{trip_id}")
def predict_path(trip_id: str):
    return _predict(trip_id)


@app.post("/predict")
def predict_body(req: PredictRequest):
    return _predict(req.trip_id)


@app.get("/")
def root():
    return {"message": "PathSnap API", "docs": "/docs", "trips": "/trips"}
