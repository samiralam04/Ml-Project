import React from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { useStore } from '../../store/useStore';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-black/80 backdrop-blur-xl border border-white/10 p-3 rounded-xl shadow-2xl">
        <p className="text-[10px] font-black tracking-widest text-slate-500 uppercase mb-2">Neural Scan</p>
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between gap-6">
            <span className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">Load</span>
            <span className="text-sm font-black text-white font-mono">{payload[0].value.toFixed(1)}%</span>
          </div>
          {payload[1] && (
            <div className="flex items-center justify-between gap-6">
              <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider">Fatigue</span>
              <span className="text-sm font-black text-white font-mono">{payload[1].value.toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export default function TemporalLoadChart() {
  const chartData = useStore((state) => state.chartData);

  return (
    <div className="w-full h-[240px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorFatigue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.2}/>
              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
          
          <XAxis 
            dataKey="time" 
            hide={true}
          />
          
          <YAxis 
            domain={[0, 100]} 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 9, fill: '#475569', fontWeight: 700 }}
            tickFormatter={(val) => `${val}%`}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          <ReferenceLine y={80} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: 'OVERLOAD', position: 'right', fill: '#f43f5e', fontSize: 8, fontWeight: 900 }} />
          <ReferenceLine y={65} stroke="#eab308" strokeDasharray="3 3" label={{ value: 'HIGH', position: 'right', fill: '#eab308', fontSize: 8, fontWeight: 900 }} />

          <Area 
            type="monotone" 
            dataKey="score" 
            stroke="#8b5cf6" 
            strokeWidth={3}
            fillOpacity={1} 
            fill="url(#colorLoad)" 
            isAnimationActive={false}
          />
          <Area 
            type="monotone" 
            dataKey="fatigue" 
            stroke="#f43f5e" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorFatigue)" 
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
