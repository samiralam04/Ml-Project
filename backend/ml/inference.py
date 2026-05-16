"""
Phase 5 — Optimized Inference Engine (replaces Phase 4 inference.py)
======================================================================
Key improvements over Phase 4:
  1. Loads config from saved checkpoint (no hardcoded params)
  2. TorchScript with graceful fallback
  3. Thread-safe via RLock (vs threading.Lock in Phase 4)
  4. Returns structured InferenceResult dataclass (typed, not raw dict)
  5. ONNX Runtime path available as alternative
  6. Cleaner session reset lifecycle
  7. All smoothing params driven by config, not hardcoded constants
"""

import os
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict
import sys

from ml.model import create_model
from ml.dataset import FeatureScaler, FEATURE_COLUMNS, SEQ_LEN
from ml.calibration import PersonalNormalizer, ProfileStore, BaselineProfile
from ml.behavioral_state import BehavioralEngine



# Typed Result
@dataclass
class InferenceResult:
    score:          float   # 0–100 display score
    raw_load:       float   # 0–1 LSTM output
    fatigue:        float   # 0–1 accumulated fatigue
    confidence:     float   # 0–1 feature quality
    buffer_fill:    float   # 0–1 temporal buffer completeness
    status:         str     # "predicting" | "warming_up" | "no_face" | "error"
    inference_ms:   float   # wall-clock latency
    state:          str     = "unknown"
    state_label:    str     = "Initializing"
    state_emoji:    str     = "⚫"
    state_color:    str     = "#6b7280"
    trend:          str     = "stable"
    is_calibrated:  bool    = False

    def to_dict(self) -> dict:
        return asdict(self)


# Adaptive Baseline
class AdaptiveBaseline:
    """Normalizes raw model output to each user's personal range."""
    def __init__(self, window: int = 300, warmup: int = 60):
        self._scores = deque(maxlen=window)
        self._warmup = warmup
        self._n      = 0

    def update(self, raw: float) -> float:
        self._scores.append(raw)
        self._n += 1
        if self._n < self._warmup or len(self._scores) < 20:
            return raw
        arr  = np.array(self._scores)
        p5   = np.percentile(arr, 5)
        p95  = np.percentile(arr, 95)
        span = p95 - p5
        if span < 0.05:
            return raw
        return float(np.clip((raw - p5) / (span + 1e-8), 0, 1))

    def reset(self):
        self._scores.clear()
        self._n = 0


# Fatigue Accumulator
class FatigueAccumulator:
    """Exponential-decay cognitive fatigue model."""
    def __init__(self, gain: float = 0.002, decay: float = 0.9995):
        self.level = 0.0
        self.gain  = gain
        self.decay = decay

    def step(self, load: float) -> float:
        if load > 0.5:
            self.level += self.gain * (load - 0.5) * 2
        self.level = float(np.clip(self.level * self.decay, 0, 1))
        return self.level

    def reset(self):
        self.level = 0.0


# CognitiveLoadEngine
class CognitiveLoadEngine:
    """
    Thread-safe real-time cognitive load inference engine.

    Lifecycle:
        1. CognitiveLoadEngine.load(checkpoint_dir)  — class method
        2. engine.predict(metrics_dict)               — called per frame
        3. engine.reset_session()                     — on new recording session
        4. engine.get_stats()                         — for /model/stats endpoint
    """

    def __init__(self, model: nn.Module, scaler: FeatureScaler, icfg: dict):
        self.model  = model
        self.model.eval()
        self.scaler = scaler
        self.icfg   = icfg

        seq_len = icfg.get("seq_len", SEQ_LEN)
        self._buffer    = deque(maxlen=seq_len)
        self._baseline  = AdaptiveBaseline(
            window  = icfg.get("baseline_window", 300),
            warmup  = icfg.get("baseline_warmup", 60),
        )
        self._fatigue   = FatigueAccumulator(
            gain  = icfg.get("fatigue_gain", 0.002),
            decay = icfg.get("fatigue_decay", 0.9995),
        )
        self._ema_score    = 50.0
        self._last_valid   = 50.0
        self._frame_count  = 0
        self._noface_count = 0
        self._latencies: deque = deque(maxlen=60)
        self._lock = threading.RLock()
        
        self._behavioral = BehavioralEngine(window_s=30.0, fps=30.0)
        self.normalizer  = PersonalNormalizer(None, global_scaler=self.scaler)

        # TorchScript compile
        self._use_script = False
        try:
            dummy = torch.randn(1, seq_len, len(FEATURE_COLUMNS))
            self._scripted = torch.jit.trace(model, dummy)
            self._use_script = True
            print("[Engine] TorchScript: ✓")
        except Exception as e:
            print(f"[Engine] TorchScript unavailable: {e}")
            self._scripted = model

    # Factory
    @classmethod
    def load(cls, checkpoint_dir: str) -> "CognitiveLoadEngine":
        ckpt_dir  = Path(checkpoint_dir)
        best_path = ckpt_dir / "best_model.pt"

        if not best_path.exists():
            raise FileNotFoundError(
                f"No trained model at {best_path}.\n"
                "Run: python ml/train.py"
            )

        state   = torch.load(str(best_path), map_location="cpu", weights_only=False)
        cfg     = state["config"]
        mcfg    = cfg["model"]
        icfg    = cfg.get("inference", {})

        model = create_model(
            arch         = "lstm",
            input_dim    = mcfg["input_dim"],
            hidden_dim   = mcfg["hidden_dim"],
            num_layers   = mcfg["num_layers"],
            dropout      = mcfg["dropout"],
            bidirectional= mcfg["bidirectional"],
            use_attention= mcfg["use_attention"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()

        scaler = FeatureScaler()
        scaler_path = ckpt_dir / "scaler.npz"
        if scaler_path.exists():
            scaler.load(str(scaler_path))
        else:
            # fallback: load global scaler from preprocess.py output
            fb_path = Path(cfg["data"]["processed_dir"]) / "scaler_params.npz"
            if fb_path.exists():
                d = np.load(str(fb_path))
                scaler.mean, scaler.std = d["mean"], d["std"]

        # Inject seq_len from config
        icfg["seq_len"] = cfg["data"]["seq_len"]

        print(f"[Engine] Loaded epoch={state['epoch']}  "
              f"val_MAE={state['val_metrics'].get('mae', '?'):.4f}")
        return cls(model, scaler, icfg)

    # Core predict()
    def predict(self, metrics: dict) -> InferenceResult:
        t0  = time.perf_counter()
        seq_len    = self.icfg.get("seq_len", SEQ_LEN)
        min_fill   = self.icfg.get("min_buffer_fill", 0.60)
        jitter_max = self.icfg.get("jitter_threshold", 5.0)
        alpha      = self.icfg.get("ema_alpha", 0.15)
        face_thr   = self.icfg.get("min_face_confidence", 0.5)

        with self._lock:
            self._frame_count += 1
            face_conf = float(metrics.get("face_confidence", 0.0))
            status    = metrics.get("status", "success")

            # Extract features
            feat_vec, feat_conf = self._extract_features(metrics)

            if face_conf < face_thr or status == "no_face":
                self._noface_count += 1
                self._buffer.append(np.zeros(len(FEATURE_COLUMNS), dtype=np.float32))
                ms = (time.perf_counter() - t0) * 1000
                return InferenceResult(self._last_valid, 0.0, self._fatigue.level,
                                       0.0, len(self._buffer)/seq_len, "no_face", ms)

            self._buffer.append(feat_vec)
            fill = len(self._buffer) / seq_len

            if fill < min_fill:
                ms = (time.perf_counter() - t0) * 1000
                return InferenceResult(self._last_valid, 0.0, self._fatigue.level,
                                       feat_conf, fill, "warming_up", ms)

            # Build sequence
            seq = np.array(list(self._buffer), dtype=np.float32)
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), len(FEATURE_COLUMNS)), dtype=np.float32)
                seq = np.vstack([pad, seq])

            # Normalize
            seq = self.normalizer.normalize(seq)

            # Inference
            with torch.no_grad():
                x        = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                raw_out  = self._scripted(x) if self._use_script else self.model(x)
                raw_load = float(raw_out.squeeze().item())

            # Scoring pipeline
            calibrated  = self._baseline.update(raw_load)
            fatigue     = self._fatigue.step(raw_load)
            blended     = calibrated * 0.85 + fatigue * 0.15
            display     = blended * 100.0

            # Anti-jitter clip
            delta = display - self._ema_score
            if abs(delta) > jitter_max:
                display = self._ema_score + (jitter_max if delta > 0 else -jitter_max)

            # EMA smoothing
            self._ema_score = alpha * display + (1 - alpha) * self._ema_score
            self._last_valid = self._ema_score

            ms = (time.perf_counter() - t0) * 1000
            self._latencies.append(ms)

            # Behavioral State
            state_info = self._behavioral.update(display, fatigue, feat_conf * face_conf, metrics.get("gaze_yaw"))

            return InferenceResult(
                score       = round(float(np.clip(self._ema_score, 0, 100)), 2),
                raw_load    = round(raw_load, 4),
                fatigue     = round(fatigue, 4),
                confidence  = round(feat_conf * face_conf, 3),
                buffer_fill = round(fill, 3),
                status      = "predicting",
                inference_ms= round(ms, 2),
                state       = state_info["state"],
                state_label = state_info["state_label"],
                state_emoji = state_info["state_emoji"],
                state_color = state_info["state_color"],
                trend       = state_info["trend"],
                is_calibrated= self.normalizer.profile is not None
            )

    def _extract_features(self, m: dict):
        vec, missing = [], 0
        for f in FEATURE_COLUMNS:
            v = m.get(f)
            if v is None or v == "NaN" or (isinstance(v, float) and np.isnan(v)):
                vec.append(0.0); missing += 1
            else:
                vec.append(float(v))
        conf = 1.0 - missing / len(FEATURE_COLUMNS)
        return np.array(vec, dtype=np.float32), conf

    # Stats & lifecycle
    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_frames":       self._frame_count,
                "no_face_frames":     self._noface_count,
                "face_rate":          round(1 - self._noface_count / max(self._frame_count, 1), 3),
                "current_score":      round(self._ema_score, 2),
                "fatigue":            round(self._fatigue.level, 3),
                "avg_inference_ms":   round(float(np.mean(list(self._latencies))) if self._latencies else 0, 2),
                "buffer_fill":        round(len(self._buffer) / self.icfg.get("seq_len", SEQ_LEN), 3),
                "session_summary":    self._behavioral.get_session_summary()
            }

    def reset_session(self, user_id: Optional[str] = None):
        with self._lock:
            self._buffer.clear()
            self._ema_score   = 50.0
            self._last_valid  = 50.0
            self._frame_count = 0
            self._noface_count= 0
            self._latencies.clear()
            self._baseline.reset()
            self._fatigue.reset()
            self._behavioral.reset()
            
            profile = None
            if user_id:
                profile = ProfileStore.load(user_id)
                if profile:
                    ProfileStore.increment_session_count(user_id)
            self.normalizer.profile = profile

        print(f"[Engine] Session reset ✓ (User: {user_id}, Calibrated: {profile is not None})")


# ONNX Export
def export_onnx(checkpoint_dir: str, output_path: str = "model.onnx") -> bool:
    try:
        import onnx, onnxruntime as ort
        engine = CognitiveLoadEngine.load(checkpoint_dir)
        model  = engine.model.eval()
        seq_len = engine.icfg.get("seq_len", SEQ_LEN)
        dummy  = torch.randn(1, seq_len, len(FEATURE_COLUMNS))

        torch.onnx.export(
            model, dummy, output_path,
            opset_version    = 17,
            do_constant_folding= True,
            input_names      = ["behavioral_features"],
            output_names     = ["cognitive_load"],
            dynamic_axes     = {"behavioral_features": {0: "batch"},
                                "cognitive_load":      {0: "batch"}},
        )
        onnx.checker.check_model(onnx.load(output_path))

        # Benchmark
        sess  = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        dummy_np = dummy.numpy()
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess.run(None, {"behavioral_features": dummy_np})
            times.append((time.perf_counter() - t0) * 1000)
        print(f"[ONNX] Exported → {output_path}")
        print(f"[ONNX] Avg latency: {np.mean(times):.2f}ms (n=100)")
        return True
    except ImportError as e:
        print(f"[ONNX] Missing: {e}. Install: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        return False


# Singleton accessor
_engine: Optional[CognitiveLoadEngine] = None

def get_engine(checkpoint_dir: Optional[str] = None) -> Optional[CognitiveLoadEngine]:
    global _engine
    if _engine is None and checkpoint_dir:
        best = Path(checkpoint_dir) / "best_model.pt"
        if best.exists():
            try:
                _engine = CognitiveLoadEngine.load(checkpoint_dir)
            except Exception as e:
                print(f"[Engine] Load error: {e}")
    return _engine
