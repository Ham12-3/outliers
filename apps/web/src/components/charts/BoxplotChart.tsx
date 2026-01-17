'use client';

import type { BoxplotData } from '@/lib/api';

interface BoxplotChartProps {
  data: BoxplotData[];
}

export function BoxplotChart({ data }: BoxplotChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {data.map((item) => (
        <BoxplotItem key={item.metric} data={item} />
      ))}
    </div>
  );
}

function BoxplotItem({ data }: { data: BoxplotData }) {
  // Calculate scale
  const allValues = [data.min, data.max, ...data.outliers];
  const minVal = Math.min(...allValues, data.lower_fence);
  const maxVal = Math.max(...allValues, data.upper_fence);
  const range = maxVal - minVal || 1;

  // Scale function (0-100%)
  const scale = (value: number) => ((value - minVal) / range) * 100;

  const whiskerLeft = scale(data.lower_fence);
  const boxLeft = scale(data.q1);
  const medianPos = scale(data.median);
  const boxRight = scale(data.q3);
  const whiskerRight = scale(data.upper_fence);

  return (
    <div>
      <h4 className="text-sm font-medium text-gray-700 mb-3 capitalize">
        {data.metric.replace('_', ' ')}
      </h4>

      {/* Boxplot visualisation */}
      <div className="relative h-12 bg-gray-100 rounded">
        {/* Whisker line */}
        <div
          className="absolute top-1/2 h-0.5 bg-gray-400 -translate-y-1/2"
          style={{
            left: `${whiskerLeft}%`,
            width: `${whiskerRight - whiskerLeft}%`,
          }}
        />

        {/* Left whisker cap */}
        <div
          className="absolute top-1/2 w-0.5 h-4 bg-gray-400 -translate-y-1/2"
          style={{ left: `${whiskerLeft}%` }}
        />

        {/* Right whisker cap */}
        <div
          className="absolute top-1/2 w-0.5 h-4 bg-gray-400 -translate-y-1/2"
          style={{ left: `${whiskerRight}%` }}
        />

        {/* Box (IQR) */}
        <div
          className="absolute top-1/2 h-8 bg-blue-200 border border-blue-400 rounded -translate-y-1/2"
          style={{
            left: `${boxLeft}%`,
            width: `${boxRight - boxLeft}%`,
          }}
        />

        {/* Median line */}
        <div
          className="absolute top-1/2 w-0.5 h-8 bg-blue-600 -translate-y-1/2"
          style={{ left: `${medianPos}%` }}
        />

        {/* Outliers */}
        {data.outliers.map((outlier, idx) => (
          <div
            key={idx}
            className="absolute top-1/2 w-3 h-3 bg-red-500 rounded-full -translate-x-1/2 -translate-y-1/2 border-2 border-white"
            style={{ left: `${scale(outlier)}%` }}
            title={`Outlier: ${outlier.toLocaleString()}`}
          />
        ))}

        {/* Fence markers (dashed lines) */}
        <div
          className="absolute top-0 w-px h-full border-l border-dashed border-gray-400"
          style={{ left: `${scale(data.lower_fence)}%` }}
        />
        <div
          className="absolute top-0 w-px h-full border-l border-dashed border-gray-400"
          style={{ left: `${scale(data.upper_fence)}%` }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between mt-2 text-xs text-gray-500">
        <span>{minVal.toLocaleString()}</span>
        <span>{maxVal.toLocaleString()}</span>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-2 mt-3 text-xs">
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-500">Q1</p>
          <p className="font-medium text-gray-900">{data.q1.toFixed(1)}</p>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-500">Median</p>
          <p className="font-medium text-gray-900">{data.median.toFixed(1)}</p>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-500">Q3</p>
          <p className="font-medium text-gray-900">{data.q3.toFixed(1)}</p>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-500">IQR</p>
          <p className="font-medium text-gray-900">{data.iqr.toFixed(1)}</p>
        </div>
      </div>

      {/* Fences */}
      <div className="flex gap-4 mt-2 text-xs">
        <div>
          <span className="text-gray-500">Lower fence: </span>
          <span className="font-medium">{data.lower_fence.toFixed(1)}</span>
        </div>
        <div>
          <span className="text-gray-500">Upper fence: </span>
          <span className="font-medium">{data.upper_fence.toFixed(1)}</span>
        </div>
        {data.outliers.length > 0 && (
          <div>
            <span className="text-red-500 font-medium">
              {data.outliers.length} outlier{data.outliers.length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
