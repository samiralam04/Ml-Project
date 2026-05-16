"""
MindFlow — FastAPI Backend (Phase 5)
=====================================
Integrates Phase 5 ML inference engine into the live WebSocket pipeline.
New in Phase 5:
  - Typed InferenceResult (not raw dict)
  - /model/eval endpoint to trigger evaluation report
  - Clean engine singleton lifecycle
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, json, asyncio, time
from pathlib import Path

import vision
from session_manager import session_manager
from ml.calibration import start_calibration, get_calibration, finish_calibration, ProfileStore


# Paths 
BASE            = Path(__file__).parent
CHECKPOINT_DIR  = str(BASE / "ml" / "checkpoints" / "phase5_bilstm_attention_v1")
PROCESSED_DIR   = str(BASE / "processed_data")
DATASET_DIR     = str(BASE / "dataset")

# Lazy engine 
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from ml.inference import get_engine as _get
        _engine = _get(CHECKPOINT_DIR)
        if _engine:
            print("[FastAPI] ML engine loaded ✓")
        else:
            print("[FastAPI] No model found — heuristic fallback active.")
    return _engine


def _heuristic_score(metrics: dict) -> float:
    ear = metrics.get("ear")
    if ear and ear != "NaN":
        return round(max(0.0, min(100.0, 100 - float(ear) * 200)), 2)
    return 50.0


# App
app = FastAPI(title="MindFlow API — Phase 5")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    engine = get_engine()
    return {
        "phase": 5,
        "status": "ok",
        "ml_ready": engine is not None,
    }


@app.get("/model/info")
def model_info():
    import torch, os
    best = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not best.exists():
        return {"status": "no_model", "tip": "Run: python ml/train.py"}
    state = torch.load(str(best), map_location="cpu", weights_only=False)
    return {
        "status":       "ok",
        "epoch":        state.get("epoch"),
        "val_metrics":  state.get("val_metrics"),
        "experiment":   state.get("config", {}).get("experiment", {}),
    }


@app.get("/model/stats")
def model_stats():
    engine = get_engine()
    if engine is None:
        return {"status": "no_model"}
    return {"status": "ok", "stats": engine.get_stats()}


@app.post("/model/eval")
async def model_eval(background_tasks: BackgroundTasks):
    """Triggers evaluation report generation in background."""
    def _run_eval():
        from ml.evaluate import run_evaluation
        run_evaluation(
            checkpoint_dir = CHECKPOINT_DIR,
            processed_dir  = PROCESSED_DIR,
            dataset_dir    = DATASET_DIR,
        )
    best = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not best.exists():
        return {"status": "no_model", "tip": "Train first: python ml/train.py"}
    background_tasks.add_task(_run_eval)
    return {"status": "queued", "message": "Evaluation running in background"}


@app.post("/preprocess")
async def run_preprocess(background_tasks: BackgroundTasks):
    """Trigger preprocessing pipeline via API."""
    def _preprocess():
        import subprocess, sys
        subprocess.run([sys.executable, "preprocess.py"], cwd=str(BASE))
    background_tasks.add_task(_preprocess)
    return {"status": "queued", "message": "Preprocessing running in background"}


# Calibration Endpoints 

class CalibrationStartRequest(BaseModel):
    user_id: str

@app.post("/calibration/start")
def api_start_calibration(req: CalibrationStartRequest):
    """Starts a calibration session for the user."""
    start_calibration(req.user_id)
    return {"status": "ok", "message": f"Calibration started for {req.user_id}"}

@app.get("/calibration/status/{user_id}")
def api_calibration_status(user_id: str):
    """Gets the progress of an active calibration."""
    cap = get_calibration(user_id)
    if not cap:
        profile = ProfileStore.load(user_id)
        if profile:
            return {"status": "done", "profile": profile.to_dict()}
        return {"status": "not_found"}
    
    return {
        "status": "calibrating",
        "progress": cap.progress,
        "seconds_remaining": cap.seconds_remaining
    }

@app.post("/calibration/finish/{user_id}")
def api_finish_calibration(user_id: str):
    """Completes the calibration and saves the profile."""
    profile = finish_calibration(user_id)
    if not profile:
        raise HTTPException(status_code=400, detail="Not enough valid frames captured for calibration.")
    return {"status": "ok", "profile": profile.to_dict()}


# Replay Endpoints

@app.get("/sessions/{user_id}/{session_id}/replay")
def get_session_replay(user_id: str, session_id: str):
    """
    Loads a historical session CSV, runs it through the ML engine, 
    and returns the entire timeline for the Replay scrubber.
    """
    csv_path = Path(DATASET_DIR) / user_id / session_id / "features.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="ML Engine not loaded")

    engine.reset_session(user_id)
    
    timeline = []
    # simulate real-time progression
    for _, row in df.iterrows():
        metrics = row.to_dict()
        # Convert nan to string 'NaN' for the engine
        for k, v in metrics.items():
            if pd.isna(v):
                metrics[k] = "NaN"
                
        metrics["status"] = "success" if metrics.get("face_confidence", 0) > 0 else "no_face"
        
        ir = engine.predict(metrics)
        timeline.append({
            "time": metrics.get("timestamp"),
            "score": ir.score,
            "raw_load": ir.raw_load,
            "fatigue": ir.fatigue,
            "state": ir.state,
            "state_label": ir.state_label,
            "state_color": ir.state_color,
            "state_emoji": ir.state_emoji,
            "metrics": metrics
        })

    summary = engine.get_stats()
    return {
        "status": "ok",
        "timeline": timeline,
        "summary": summary
    }



@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    engine = get_engine()
    current_session = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"type": "FRAME", "data": raw, "timestamp": time.time()}

            msg = payload.get("type")

            if msg == "START_SESSION":
                meta = payload.get("metadata", {})
                current_session = await session_manager.start_session(meta)
                if engine:
                    engine.reset_session(meta.get("participant_id"))
                await websocket.send_text(json.dumps({
                    "type":       "SESSION_STARTED",
                    "session_id": current_session,
                    "ml_ready":   engine is not None,
                }))

            elif msg == "FRAME":
                frame_data = payload.get("data")
                ts         = payload.get("timestamp", time.time())

                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, vision.process_frame, frame_data, ts
                )
                status  = result.get("status", "error")
                metrics = result.get("metrics", {})
                metrics["status"] = status

                if current_session and status in ("success", "no_face"):
                    await session_manager.log_frame(current_session, metrics)

                # Route frame to calibration if active
                meta = payload.get("metadata", {})
                participant_id = meta.get("participant_id")
                if participant_id:
                    cap = get_calibration(participant_id)
                    if cap and not cap.done and status == "success":
                        cap.add_frame(metrics)

                # Score
                if engine:
                    ir = await loop.run_in_executor(None, engine.predict, metrics)
                    score    = ir.score
                    ml_data  = ir.to_dict()
                else:
                    score   = _heuristic_score(metrics)
                    ml_data = {"status": "heuristic"}

                await websocket.send_text(json.dumps({
                    "type":    "score",
                    "score":   score,
                    "status":  status,
                    "metrics": metrics,
                    "ml":      ml_data,
                }))

            elif msg == "STOP_SESSION":
                if current_session:
                    end_meta = payload.get("end_metadata", {})
                    if engine:
                        end_meta["ml_stats"] = engine.get_stats()
                    await session_manager.stop_session(current_session, end_meta)
                    await websocket.send_text(json.dumps({
                        "type":       "SESSION_STOPPED",
                        "session_id": current_session,
                        "summary":    end_meta.get("ml_stats")
                    }))
                    current_session = None

    except WebSocketDisconnect:
        print("WebSocket disconnected")
        if current_session:
            await session_manager.stop_session(
                current_session, {"status": "disconnected"}
            )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
