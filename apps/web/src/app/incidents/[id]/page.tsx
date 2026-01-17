'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { format, parseISO } from 'date-fns';
import { clsx } from 'clsx';
import { api, type Incident, type TimeSeriesData, type BoxplotData } from '@/lib/api';
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart';
import { BoxplotChart } from '@/components/charts/BoxplotChart';
import { DetectionExplanation } from '@/components/DetectionExplanation';
import {
  ArrowLeft,
  User,
  Calendar,
  Store,
  Package,
  AlertTriangle,
  CheckCircle,
  Clock,
  MessageSquare,
} from 'lucide-react';

function getSeverityInfo(score: number) {
  if (score >= 80) return { label: 'Critical', color: 'text-red-600', bg: 'bg-red-100' };
  if (score >= 60) return { label: 'High', color: 'text-orange-600', bg: 'bg-orange-100' };
  if (score >= 30) return { label: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-100' };
  return { label: 'Low', color: 'text-green-600', bg: 'bg-green-100' };
}

export default function IncidentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const incidentId = parseInt(params.id as string);

  const [incident, setIncident] = useState<Incident | null>(null);
  const [detectionDetails, setDetectionDetails] = useState<any>(null);
  const [timeseries, setTimeseries] = useState<TimeSeriesData[]>([]);
  const [boxplot, setBoxplot] = useState<BoxplotData[]>([]);
  const [selectedSku, setSelectedSku] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [newNote, setNewNote] = useState('');
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [incidentData, detailsData] = await Promise.all([
          api.getIncident(incidentId),
          api.getIncidentDetectionDetails(incidentId),
        ]);
        setIncident(incidentData);
        setDetectionDetails(detailsData);

        // Select first SKU by default
        if (incidentData.items && incidentData.items.length > 0) {
          setSelectedSku(incidentData.items[0].sku_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load incident');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [incidentId]);

  useEffect(() => {
    async function fetchSkuData() {
      if (!incident || !selectedSku) return;

      try {
        const [timeseriesData, boxplotData] = await Promise.all([
          api.getTimeseries(incident.store_id, selectedSku),
          api.getBoxplot(incident.store_id, selectedSku),
        ]);
        setTimeseries(timeseriesData);
        setBoxplot(boxplotData);
      } catch (err) {
        console.error('Failed to load SKU data:', err);
      }
    }
    fetchSkuData();
  }, [incident, selectedSku]);

  const handleStatusChange = async (newStatus: string) => {
    if (!incident) return;
    setUpdating(true);
    try {
      const updated = await api.updateIncident(incidentId, { status: newStatus });
      setIncident(updated);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to update status');
    } finally {
      setUpdating(false);
    }
  };

  const handleAddNote = async () => {
    if (!newNote.trim()) return;
    setUpdating(true);
    try {
      await api.addIncidentNote(incidentId, newNote);
      const updated = await api.getIncident(incidentId);
      setIncident(updated);
      setNewNote('');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to add note');
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (error || !incident) {
    return (
      <div className="space-y-4">
        <Link href="/incidents" className="flex items-center gap-2 text-primary-600 hover:text-primary-700">
          <ArrowLeft className="h-4 w-4" />
          Back to incidents
        </Link>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error || 'Incident not found'}
        </div>
      </div>
    );
  }

  const severity = getSeverityInfo(incident.severity_score);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link href="/incidents" className="flex items-center gap-2 text-primary-600 hover:text-primary-700 text-sm mb-2">
            <ArrowLeft className="h-4 w-4" />
            Back to incidents
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">{incident.headline}</h1>
          <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <Store className="h-4 w-4" />
              {incident.store_id}
            </span>
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              {format(parseISO(incident.date), 'dd MMMM yyyy')}
            </span>
            <span className="flex items-center gap-1">
              <Package className="h-4 w-4" />
              {incident.sku_count} SKU{incident.sku_count !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
        <div className={clsx('px-4 py-2 rounded-lg', severity.bg)}>
          <p className="text-xs text-gray-500">Severity</p>
          <p className={clsx('text-xl font-bold', severity.color)}>
            {incident.severity_score.toFixed(0)} - {severity.label}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Description */}
          {incident.description && (
            <div className="card p-4">
              <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
              <p className="text-gray-600 whitespace-pre-line">{incident.description}</p>
            </div>
          )}

          {/* SKU selector */}
          {incident.items && incident.items.length > 1 && (
            <div className="card p-4">
              <h3 className="font-semibold text-gray-900 mb-2">Affected SKUs</h3>
              <div className="flex flex-wrap gap-2">
                {incident.items.map((item) => (
                  <button
                    key={item.sku_id}
                    onClick={() => setSelectedSku(item.sku_id)}
                    className={clsx(
                      'px-3 py-1 rounded-full text-sm font-medium transition-colors',
                      selectedSku === item.sku_id
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    )}
                  >
                    {item.sku_id}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Time series charts */}
          {selectedSku && timeseries.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold text-gray-900 mb-4">
                Time Series - {selectedSku}
              </h3>
              <div className="space-y-6">
                {timeseries.map((series) => (
                  <div key={series.metric}>
                    <h4 className="text-sm font-medium text-gray-700 mb-2 capitalize">
                      {series.metric.replace('_', ' ')}
                    </h4>
                    <TimeSeriesChart data={series.data} metric={series.metric} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Boxplot charts */}
          {selectedSku && boxplot.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold text-gray-900 mb-4">
                Distribution Analysis - {selectedSku}
              </h3>
              <BoxplotChart data={boxplot} />
            </div>
          )}

          {/* Detection explanations */}
          {selectedSku && detectionDetails && (
            <DetectionExplanation
              skuId={selectedSku}
              detections={detectionDetails.detections_by_sku[selectedSku]}
            />
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Status panel */}
          <div className="card p-4">
            <h3 className="font-semibold text-gray-900 mb-4">Status</h3>
            <div className="space-y-3">
              <button
                onClick={() => handleStatusChange('open')}
                disabled={updating}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-colors',
                  incident.status === 'open'
                    ? 'bg-red-100 text-red-700 ring-2 ring-red-500'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                <AlertTriangle className="h-4 w-4" />
                Open
              </button>
              <button
                onClick={() => handleStatusChange('investigating')}
                disabled={updating}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-colors',
                  incident.status === 'investigating'
                    ? 'bg-yellow-100 text-yellow-700 ring-2 ring-yellow-500'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                <Clock className="h-4 w-4" />
                Investigating
              </button>
              <button
                onClick={() => handleStatusChange('resolved')}
                disabled={updating}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-colors',
                  incident.status === 'resolved'
                    ? 'bg-green-100 text-green-700 ring-2 ring-green-500'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                <CheckCircle className="h-4 w-4" />
                Resolved
              </button>
            </div>

            {/* Assignee */}
            <div className="mt-4 pt-4 border-t border-gray-200">
              <label className="label">Assignee</label>
              <div className="flex items-center gap-2">
                <User className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">
                  {incident.assignee || 'Unassigned'}
                </span>
              </div>
            </div>

            {/* Estimated impact */}
            {incident.estimated_impact && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-500">Estimated Impact</p>
                <p className="text-lg font-semibold text-gray-900">
                  £{incident.estimated_impact.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
            )}
          </div>

          {/* Notes */}
          <div className="card p-4">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Notes
            </h3>

            {/* Add note */}
            <div className="mb-4">
              <textarea
                className="input resize-none"
                rows={3}
                placeholder="Add a note..."
                value={newNote}
                onChange={(e) => setNewNote(e.target.value)}
              />
              <button
                className="btn-primary w-full mt-2"
                onClick={handleAddNote}
                disabled={!newNote.trim() || updating}
              >
                Add Note
              </button>
            </div>

            {/* Notes list */}
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {incident.notes?.map((note) => (
                <div
                  key={note.id}
                  className={clsx(
                    'p-3 rounded-lg text-sm',
                    note.note_type === 'comment'
                      ? 'bg-gray-50'
                      : 'bg-blue-50'
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-900">
                      {note.author}
                    </span>
                    <span className="text-xs text-gray-500">
                      {format(parseISO(note.created_at), 'dd MMM HH:mm')}
                    </span>
                  </div>
                  <p className="text-gray-600">{note.content}</p>
                </div>
              ))}
              {(!incident.notes || incident.notes.length === 0) && (
                <p className="text-gray-500 text-sm text-center py-4">
                  No notes yet
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
