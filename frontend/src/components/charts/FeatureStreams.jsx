import React from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../../store/useStore';
import { 
  Eye, 
  User, 
  RotateCw, 
  Sun, 
  Target,
  BarChart
} from 'lucide-react';

function TelemetryRow({ label, value, icon: Icon, format = "0.0", color = "from-blue-500 to-indigo-500", max = 1 }) {
  const raw = (value === "NaN" || value == null) ? null : Number(value);
  const display = raw == null ? "STANDBY" : raw.toFixed((format.split('.')[1] || '').length || 1);
  const pct     = raw == null ? 0   : Math.min(100, (Math.abs(raw) / max) * 100);
  
  return (
    <div className="group flex flex-col gap-2 p-3 rounded-xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-all duration-300">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {Icon && <Icon size={12} className="text-slate-300 group-hover:text-slate-100 transition-colors" />}
          <span className="text-[9px] font-black tracking-widest text-slate-300 uppercase">{label}</span>
        </div>
        <span className="text-[10px] font-mono font-black text-slate-200 tracking-wider">{display}</span>
      </div>
      <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          className={`h-full bg-gradient-to-r ${color} shadow-[0_0_8px_rgba(255,255,255,0.1)]`}
        />
      </div>
    </div>
  );
}

export default function FeatureStreams() {
  const liveMetrics = useStore((state) => state.liveMetrics);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 mb-2">
        <BarChart size={14} className="text-violet-400" />
        <h3 className="text-[10px] font-black tracking-[0.2em] uppercase text-slate-400">Biological Telemetry</h3>
      </div>
      
      <div className="grid grid-cols-2 gap-3">
        <TelemetryRow label="Eye Aperture" value={liveMetrics.ear} icon={Eye} format="0.00" color="from-cyan-500 to-blue-500" max={0.4} />
        <TelemetryRow label="Brow Tension" value={liveMetrics.eyebrow_tension} icon={Target} format="0.00" color="from-violet-500 to-purple-500" max={0.5} />
        <TelemetryRow label="Optic Openness" value={liveMetrics.eye_openness} icon={Eye} format="0.00" color="from-blue-400 to-indigo-600" />
        <TelemetryRow label="Lumen Density" value={liveMetrics.light_intensity} icon={Sun} format="0.0" color="from-amber-400 to-orange-500" max={255} />
      </div>

      <div className="h-px bg-white/5 my-2" />

      <div className="grid grid-cols-3 gap-3">
        <TelemetryRow label="Pitch" value={liveMetrics.head_pitch} icon={RotateCw} format="0.1" color="from-slate-500 to-slate-300" max={45} />
        <TelemetryRow label="Yaw" value={liveMetrics.head_yaw} icon={RotateCw} format="0.1" color="from-slate-500 to-slate-300" max={45} />
        <TelemetryRow label="Roll" value={liveMetrics.head_roll} icon={RotateCw} format="0.1" color="from-slate-500 to-slate-300" max={45} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <TelemetryRow label="Gaze Latitude" value={liveMetrics.gaze_pitch} icon={Target} format="0.3" color="from-emerald-500 to-teal-500" max={0.5} />
        <TelemetryRow label="Gaze Longitude" value={liveMetrics.gaze_yaw} icon={Target} format="0.3" color="from-emerald-500 to-teal-500" max={0.5} />
      </div>
    </div>
  );
}
