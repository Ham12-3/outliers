'use client';

import { clsx } from 'clsx';
import { AlertTriangle, TrendingUp, TrendingDown, Info } from 'lucide-react';

interface TukeyDetection {
  metric: string;
  actual_value: number;
  q1: number;
  q3: number;
  iqr: number;
  lower_fence: number;
  upper_fence: number;
  outlier_distance: number;
  sample_size: number;
  fallback_used?: string;
}

interface IsolationForestDetection {
  anomaly_score: number;
  threshold: number;
  reasons: Array<{
    feature: string;
    z_score: number;
    message: string;
    value: number;
    mean: number;
  }>;
  fallback_used?: string;
}

interface DetectionExplanationProps {
  skuId: string;
  detections?: {
    tukey: TukeyDetection[];
    isolation_forest?: IsolationForestDetection;
  };
}

export function DetectionExplanation({ skuId, detections }: DetectionExplanationProps) {
  if (!detections) {
    return null;
  }

  return (
    <div className="card p-4">
      <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <AlertTriangle className="h-5 w-5 text-yellow-500" />
        Why was this flagged? - {skuId}
      </h3>

      <div className="space-y-6">
        {/* Tukey explanations */}
        {detections.tukey && detections.tukey.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
              <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs font-semibold">
                Tukey IQR
              </span>
              Statistical Outlier Detection
            </h4>

            <div className="space-y-3">
              {detections.tukey.map((detection, idx) => {
                const isAbove = detection.outlier_distance > 0;
                const Icon = isAbove ? TrendingUp : TrendingDown;

                return (
                  <div
                    key={idx}
                    className="bg-gray-50 rounded-lg p-3 border border-gray-200"
                  >
                    <div className="flex items-start gap-3">
                      <div className={clsx(
                        'p-1.5 rounded',
                        isAbove ? 'bg-red-100' : 'bg-blue-100'
                      )}>
                        <Icon className={clsx(
                          'h-4 w-4',
                          isAbove ? 'text-red-600' : 'text-blue-600'
                        )} />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-gray-900 capitalize">
                          {detection.metric.replace('_', ' ')}
                        </p>
                        <p className="text-sm text-gray-600 mt-1">
                          Actual value: <span className="font-semibold">{detection.actual_value.toLocaleString()}</span>
                        </p>
                        <p className="text-sm text-gray-600">
                          {isAbove ? (
                            <>
                              <span className="text-red-600 font-medium">
                                {Math.abs(detection.outlier_distance).toFixed(1)} above
                              </span>
                              {' '}the upper fence ({detection.upper_fence.toFixed(1)})
                            </>
                          ) : (
                            <>
                              <span className="text-blue-600 font-medium">
                                {Math.abs(detection.outlier_distance).toFixed(1)} below
                              </span>
                              {' '}the lower fence ({detection.lower_fence.toFixed(1)})
                            </>
                          )}
                        </p>
                        <div className="text-xs text-gray-500 mt-2">
                          Based on {detection.sample_size} samples
                          {detection.fallback_used && (
                            <span className="ml-2 bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">
                              Used {detection.fallback_used} fallback
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Isolation Forest explanation */}
        {detections.isolation_forest && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
              <span className="bg-purple-100 text-purple-700 px-2 py-0.5 rounded text-xs font-semibold">
                Isolation Forest
              </span>
              Machine Learning Detection
            </h4>

            <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-gray-600">Anomaly Score</span>
                <span className={clsx(
                  'font-semibold',
                  detections.isolation_forest.anomaly_score < 0 ? 'text-red-600' : 'text-gray-900'
                )}>
                  {detections.isolation_forest.anomaly_score.toFixed(3)}
                </span>
              </div>

              {/* Anomaly score bar */}
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden mb-4">
                <div
                  className={clsx(
                    'h-full transition-all',
                    detections.isolation_forest.anomaly_score < 0
                      ? 'bg-red-500'
                      : 'bg-green-500'
                  )}
                  style={{
                    width: `${Math.abs(detections.isolation_forest.anomaly_score * 50 + 50)}%`,
                  }}
                />
              </div>

              {/* Contributing features */}
              {detections.isolation_forest.reasons && detections.isolation_forest.reasons.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase mb-2">
                    Top Contributing Features
                  </p>
                  <div className="space-y-2">
                    {detections.isolation_forest.reasons.map((reason, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-white rounded p-2 border border-gray-100"
                      >
                        <div>
                          <span className="text-sm font-medium text-gray-900 capitalize">
                            {reason.feature.replace('_', ' ')}
                          </span>
                          <p className="text-xs text-gray-500">
                            {reason.message}
                          </p>
                        </div>
                        <div className={clsx(
                          'text-sm font-semibold',
                          Math.abs(reason.z_score) > 2 ? 'text-red-600' : 'text-yellow-600'
                        )}>
                          {reason.z_score > 0 ? '+' : ''}{reason.z_score.toFixed(1)}σ
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {detections.isolation_forest.fallback_used && (
                <div className="text-xs text-gray-500 mt-2">
                  <span className="bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">
                    Used {detections.isolation_forest.fallback_used} model
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* No detections */}
        {(!detections.tukey || detections.tukey.length === 0) && !detections.isolation_forest && (
          <div className="flex items-center gap-2 text-gray-500 bg-gray-50 rounded-lg p-4">
            <Info className="h-5 w-5" />
            <p>No detection details available for this SKU</p>
          </div>
        )}
      </div>
    </div>
  );
}
