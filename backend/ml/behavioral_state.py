"""
Phase 6 — Behavioral Intelligence Engine
==========================================
Converts raw LSTM load scores into interpretable behavioral states.

Inspired by cognitive science models:
  - Yerkes-Dodson inverted-U (performance vs. arousal)
  - Attention Restoration Theory (recovery after overload)
  - NASA-TLX dimensions (mental demand, temporal demand, frustration)

State Machine:
  ┌────────────┐   load < 30   ┌────────────┐
  │  FOCUSED   │◄──────────────│  RELAXED   │
  │ 30–65 load │               │  0–30 load │
  └─────┬──────┘               └────────────┘
        │ load > 65, duration > 5s    ▲
        ▼                             │ load < 40, duration > 8s
  ┌────────────┐   load > 80   ┌─────┴──────┐
  │  ELEVATED  │──────────────►│ RECOVERING │
  │ 65–80 load │               │ decreasing │
  └─────┬──────┘               └────────────┘
        │ load > 80, duration > 8s
        ▼
  ┌────────────┐   fatigue > 0.3   ┌────────────┐
  │ OVERLOADED │──────────────────►│  FATIGUED  │
  │  80+ load  │                   │ high+tired │
  └────────────┘                   └────────────┘
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, List


# State Definitions
class BehavioralState(str, Enum):
    UNKNOWN     = "unknown"
    CALIBRATING = "calibrating"
    RELAXED     = "relaxed"
    FOCUSED     = "focused"
    ELEVATED    = "elevated"
    OVERLOADED  = "overloaded"
    FATIGUED    = "fatigued"
    RECOVERING  = "recovering"
    DISTRACTED  = "distracted"


STATE_META = {
    BehavioralState.UNKNOWN:     {"emoji": "⚫", "color": "#6b7280", "label": "Initializing"},
    BehavioralState.CALIBRATING: {"emoji": "⚙️",  "color": "#8b5cf6", "label": "Calibrating"},
    BehavioralState.RELAXED:     {"emoji": "🟢", "color": "#22c55e", "label": "Relaxed"},
    BehavioralState.FOCUSED:     {"emoji": "🔵", "color": "#3b82f6", "label": "Focused"},
    BehavioralState.ELEVATED:    {"emoji": "🟡", "color": "#eab308", "label": "Elevated Load"},
    BehavioralState.OVERLOADED:  {"emoji": "🔴", "color": "#ef4444", "label": "Overloaded"},
    BehavioralState.FATIGUED:    {"emoji": "🟣", "color": "#a855f7", "label": "Fatigued"},
    BehavioralState.RECOVERING:  {"emoji": "🩵", "color": "#06b6d4", "label": "Recovering"},
    BehavioralState.DISTRACTED:  {"emoji": "🟠", "color": "#f97316", "label": "Distracted"},
}


# State Event
@dataclass
class StateEvent:
    state:      str
    timestamp:  float
    score:      float
    fatigue:    float
    duration_s: float   # how long previous state lasted


# Behavioral Intelligence Engine
class BehavioralEngine:
    """
    Stateful engine that converts per-frame (score, fatigue, confidence)
    into interpretable behavioral states with hysteresis and trend analysis.

    Hysteresis prevents rapid oscillation between states by requiring
    a score to stay in a new regime for a minimum number of seconds
    before transitioning. This mirrors how cognitive state changes
    are gradual in reality.
    """

    # State transition thresholds
    RELAXED_THRESH  = 30.0   # score < 30 → relaxed
    FOCUSED_THRESH  = 65.0   # 30–65 → focused
    ELEVATED_THRESH = 80.0   # 65–80 → elevated
    # > 80 → overloaded

    FATIGUE_THRESH  = 0.30   # fatigue > 0.30 → fatigued (overrides overloaded)

    # Hysteresis: require this many seconds in a regime before state switch
    MIN_DWELL_S = {
        BehavioralState.RELAXED:    5.0,
        BehavioralState.FOCUSED:    3.0,
        BehavioralState.ELEVATED:   5.0,
        BehavioralState.OVERLOADED: 6.0,
        BehavioralState.FATIGUED:   8.0,
        BehavioralState.RECOVERING: 8.0,
        BehavioralState.DISTRACTED: 4.0,
    }

    def __init__(self, window_s: float = 30.0, fps: float = 30.0):
        self._fps        = fps
        self._win_n      = int(window_s * fps)

        # Score history for trend analysis
        self._scores:   deque = deque(maxlen=self._win_n)
        self._fatigues: deque = deque(maxlen=self._win_n)
        self._times:    deque = deque(maxlen=self._win_n)

        # Head gaze history for distraction detection
        self._gaze_yaws: deque = deque(maxlen=90)   # 3s

        # Current state machine
        self.state          = BehavioralState.UNKNOWN
        self._regime_start  = time.time()
        self._regime_score  = 50.0
        self._state_entered = time.time()

        # Session history
        self.state_history: List[StateEvent] = []
        self._overload_total_s = 0.0
        self._recovery_start: Optional[float] = None

    # Main update
    def update(self, score: float, fatigue: float, confidence: float,
               gaze_yaw: Optional[float] = None) -> Dict:
        now = time.time()
        self._scores.append(score)
        self._fatigues.append(fatigue)
        self._times.append(now)
        if gaze_yaw is not None:
            self._gaze_yaws.append(abs(float(gaze_yaw)))

        # Determine target regime
        target = self._classify(score, fatigue, confidence, gaze_yaw)

        # Hysteresis: only switch if we've been in this regime for min_dwell
        if target != self.state:
            if self._regime_score != target.value:
                self._regime_start = now
                self._regime_score = target.value
            dwell = now - self._regime_start
            min_dwell = self.MIN_DWELL_S.get(target, 3.0)
            if dwell >= min_dwell:
                self._transition_to(target, score, fatigue, now)
        else:
            self._regime_start = now
            self._regime_score = target.value

        # Track overload duration
        if self.state == BehavioralState.OVERLOADED:
            self._overload_total_s += 1.0 / max(self._fps, 1)

        meta = STATE_META.get(self.state, STATE_META[BehavioralState.UNKNOWN])
        trend = self._trend()
        return {
            "state":              self.state.value,
            "state_label":        meta["label"],
            "state_emoji":        meta["emoji"],
            "state_color":        meta["color"],
            "trend":              trend,             # "rising" | "stable" | "falling"
            "overload_total_s":   round(self._overload_total_s, 1),
            "state_duration_s":   round(now - self._state_entered, 1),
            "session_avg_score":  round(float(np.mean(list(self._scores))), 1) if self._scores else 50.0,
            "session_peak_score": round(float(max(self._scores)), 1) if self._scores else 50.0,
        }

    def _classify(self, score: float, fatigue: float, confidence: float,
                  gaze_yaw: Optional[float]) -> BehavioralState:
        """Rule-based state classifier with multi-factor logic."""

        # Distraction: extreme sustained gaze deviation
        if (len(self._gaze_yaws) >= 30 and
                float(np.mean(list(self._gaze_yaws)[-30:])) > 0.35):
            return BehavioralState.DISTRACTED

        # Fatigue overrides everything at high sustained load
        if fatigue > self.FATIGUE_THRESH and score > 70:
            return BehavioralState.FATIGUED

        # Recovery: was overloaded/fatigued, now dropping
        if self.state in (BehavioralState.OVERLOADED, BehavioralState.FATIGUED,
                          BehavioralState.ELEVATED):
            if score < self.FOCUSED_THRESH and self._trend() == "falling":
                return BehavioralState.RECOVERING

        # Primary score-based classification
        if score >= self.ELEVATED_THRESH:
            return BehavioralState.OVERLOADED
        if score >= self.FOCUSED_THRESH:
            return BehavioralState.ELEVATED
        if score >= self.RELAXED_THRESH:
            return BehavioralState.FOCUSED
        return BehavioralState.RELAXED

    def _transition_to(self, new_state: BehavioralState, score: float,
                        fatigue: float, now: float):
        """Record state transition and reset timers."""
        if self.state != BehavioralState.UNKNOWN:
            event = StateEvent(
                state     = self.state.value,
                timestamp = self._state_entered,
                score     = score,
                fatigue   = fatigue,
                duration_s= now - self._state_entered,
            )
            self.state_history.append(event)

        self.state          = new_state
        self._state_entered = now
        self._regime_start  = now

    def _trend(self) -> str:
        """Compute recent score trend over last 10 seconds."""
        if len(self._scores) < 20:
            return "stable"
        recent  = list(self._scores)[-90:]   # last 3s
        earlier = list(self._scores)[-180:-90] if len(self._scores) >= 180 else list(self._scores)[:90]
        if not earlier:
            return "stable"
        delta = float(np.mean(recent)) - float(np.mean(earlier))
        if delta > 4.0:
            return "rising"
        if delta < -4.0:
            return "falling"
        return "stable"

    def get_session_summary(self) -> Dict:
        """Full session analytics summary."""
        scores  = list(self._scores)
        if not scores:
            return {}
        state_counts: Dict[str, float] = {}
        for ev in self.state_history:
            state_counts[ev.state] = state_counts.get(ev.state, 0) + ev.duration_s

        # Find most frequent state
        most_frequent = "relaxed"
        if state_counts:
            most_frequent = max(state_counts, key=state_counts.get)

        # Calculate state percentages
        total_s = sum(state_counts.values()) or 1.0
        state_pct = {k: round((v / total_s) * 100, 1) for k, v in state_counts.items()}

        return {
            "mean_load":         round(float(np.mean(scores)), 1),
            "peak_load":         round(float(np.max(scores)), 1),
            "min_load":          round(float(np.min(scores)), 1),
            "std_load":          round(float(np.std(scores)), 2),
            "most_frequent_state": most_frequent,
            "overload_total_s":  round(self._overload_total_s, 1),
            "state_breakdown_s": {k: round(v, 1) for k, v in state_counts.items()},
            "state_percentages": state_pct,
            "n_state_transitions": len(self.state_history),
            "state_history":     [asdict(e) for e in self.state_history[-50:]],
        }

    def reset(self):
        self._scores.clear()
        self._fatigues.clear()
        self._times.clear()
        self._gaze_yaws.clear()
        self.state            = BehavioralState.UNKNOWN
        self._regime_start    = time.time()
        self._state_entered   = time.time()
        self.state_history    = []
        self._overload_total_s = 0.0
