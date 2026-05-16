import React from 'react';
import { useStore } from '../../store/useStore';
import { motion } from 'framer-motion';

const STATE_COLORS = {
  relaxed: '#10b981',
  focused: '#3b82f6',
  elevated: '#f59e0b',
  overloaded: '#ef4444',
  fatigued: '#8b5cf6',
  recovering: '#06b6d4',
  distracted: '#f97316',
  unknown: '#475569'
};

export default function BehavioralTimeline() {
  const data = useStore((state) => state.chartData);
  
  if (data.length < 2) {
    return (
      <div className="w-full h-full flex flex-col justify-center">
        <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden" />
      </div>
    );
  }

  const startTime = data[0].time;
  const endTime = data[data.length - 1].time;
  const totalDuration = endTime - startTime || 1;

  // Segment contiguous states
  const segments = [];
  let currentSegment = null;

  data.forEach((point) => {
    if (!currentSegment || currentSegment.state !== point.state) {
      if (currentSegment) {
        currentSegment.end = point.time;
        segments.push(currentSegment);
      }
      currentSegment = { state: point.state, start: point.time, end: point.time };
    } else {
      currentSegment.end = point.time;
    }
  });
  if (currentSegment) {
    currentSegment.end = endTime;
    segments.push(currentSegment);
  }

  return (
    <div className="w-full h-12 flex flex-col justify-center py-2">
      <div className="w-full h-4 flex items-center relative rounded-full overflow-hidden bg-white/5 border border-white/5">
        {segments.map((seg, i) => {
          const widthPct = Math.max(0.5, ((seg.end - seg.start) / totalDuration) * 100);
          return (
            <motion.div 
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="h-full relative group"
              style={{ 
                width: `${widthPct}%`, 
                backgroundColor: STATE_COLORS[seg.state] || STATE_COLORS.unknown,
              }}
            >
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black/90 border border-white/10 rounded text-[8px] font-black uppercase text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none">
                {seg.state}
              </div>
            </motion.div>
          );
        })}
      </div>
      <div className="flex justify-between mt-2 px-1">
        <span className="text-[8px] font-black text-slate-500 tracking-widest uppercase">Start Session</span>
        <span className="text-[8px] font-black text-slate-500 tracking-widest uppercase">T - {(totalDuration / 1000).toFixed(0)}S</span>
      </div>
    </div>
  );
}
