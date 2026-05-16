import React, { useState, useEffect, useRef } from 'react';
import { useStore } from '../../store/useStore';
import { Play, Pause, FastForward, X } from 'lucide-react';

export default function ReplayControls({ sessionData, onClose }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentIndex, setCurrentIndex] = useState(0);
  const timerRef = useRef(null);
  const addFrameData = useStore(state => state.addFrameData);
  const startSession = useStore(state => state.startSession);
  const stopSession = useStore(state => state.stopSession);

  const totalFrames = sessionData.timeline.length;

  useEffect(() => {
    // Init store for replay
    startSession('replay_session', true);
    return () => stopSession(sessionData.summary);
  }, []);

  useEffect(() => {
    if (isPlaying && currentIndex < totalFrames) {
      timerRef.current = setInterval(() => {
        setCurrentIndex(prev => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false);
            return prev;
          }
          const frame = sessionData.timeline[prev];
          // Mock the websocket payload shape
          addFrameData({
            score: frame.score,
            ml: {
              raw_load: frame.raw_load,
              fatigue: frame.fatigue,
              state: frame.state,
              state_label: frame.state_label,
              state_color: frame.state_color,
              state_emoji: frame.state_emoji,
              status: "predicting",
            },
            metrics: frame.metrics
          });
          return prev + 1;
        });
      }, 33 / speed); // 30fps default
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, speed, currentIndex, totalFrames, sessionData, addFrameData]);

  const handleScrub = (e) => {
    const pct = parseFloat(e.target.value);
    const targetIdx = Math.floor((pct / 100) * totalFrames);
    setCurrentIndex(targetIdx);
  };

  return (
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 w-[600px] bg-gray-900/95 backdrop-blur-xl border border-gray-700 rounded-2xl shadow-2xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-violet-500 animate-pulse" />
          <span className="text-xs font-bold tracking-widest text-violet-400">SESSION REPLAY</span>
        </div>
        <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={16}/></button>
      </div>

      {/* Scrubber */}
      <div className="w-full flex items-center gap-3">
        <span className="text-xs text-gray-500 font-mono">{(currentIndex / 30).toFixed(1)}s</span>
        <input 
          type="range" 
          min="0" max="100" 
          value={(currentIndex / Math.max(1, totalFrames)) * 100}
          onChange={handleScrub}
          className="flex-1 accent-violet-500 h-1.5 bg-gray-800 rounded-full appearance-none cursor-pointer"
        />
        <span className="text-xs text-gray-500 font-mono">{(totalFrames / 30).toFixed(1)}s</span>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="w-10 h-10 rounded-full bg-violet-600 hover:bg-violet-500 flex items-center justify-center shadow-[0_0_15px_rgba(139,92,246,0.4)] transition"
        >
          {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" className="ml-1" />}
        </button>
        <button 
          onClick={() => setSpeed(s => s >= 4 ? 1 : s * 2)}
          className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-mono font-bold flex items-center gap-1 transition"
        >
          <FastForward size={14} /> {speed}x
        </button>
      </div>
    </div>
  );
}
