'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import type { TimeSeriesPoint } from '@/lib/api';

interface TimeSeriesChartProps {
  data: TimeSeriesPoint[];
  metric: string;
}

export function TimeSeriesChart({ data, metric }: TimeSeriesChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  const outliers = data.filter((d) => d.is_outlier);

  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            tickFormatter={(value) => format(parseISO(value), 'dd MMM')}
            stroke="#6b7280"
            fontSize={11}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#6b7280"
            fontSize={11}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <Tooltip
            labelFormatter={(value) => format(parseISO(value as string), 'dd MMM yyyy')}
            formatter={(value: number) => [value.toLocaleString(), metric.replace('_', ' ')]}
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem',
              fontSize: '12px',
            }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          {/* Highlight outliers */}
          {outliers.map((outlier, idx) => (
            <ReferenceDot
              key={idx}
              x={outlier.date}
              y={outlier.value}
              r={6}
              fill="#ef4444"
              stroke="#fff"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
