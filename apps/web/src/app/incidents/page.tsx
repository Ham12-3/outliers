'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { format, parseISO } from 'date-fns';
import { clsx } from 'clsx';
import { api, type Incident } from '@/lib/api';
import { Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react';

function getSeverityBadge(score: number) {
  if (score >= 80) return { label: 'Critical', class: 'badge-danger' };
  if (score >= 60) return { label: 'High', class: 'bg-orange-100 text-orange-800' };
  if (score >= 30) return { label: 'Medium', class: 'badge-warning' };
  return { label: 'Low', class: 'badge-success' };
}

function getStatusBadge(status: string) {
  switch (status) {
    case 'open':
      return { label: 'Open', class: 'badge-danger' };
    case 'investigating':
      return { label: 'Investigating', class: 'badge-warning' };
    case 'resolved':
      return { label: 'Resolved', class: 'badge-success' };
    default:
      return { label: status, class: 'badge-info' };
  }
}

export default function IncidentsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stores, setStores] = useState<string[]>([]);

  const page = parseInt(searchParams.get('page') || '1');
  const status = searchParams.get('status') || '';
  const store = searchParams.get('store') || '';
  const pageSize = 20;

  useEffect(() => {
    async function fetchStores() {
      try {
        const result = await api.getStores();
        setStores(result.stores);
      } catch (err) {
        console.error('Failed to load stores:', err);
      }
    }
    fetchStores();
  }, []);

  useEffect(() => {
    async function fetchIncidents() {
      try {
        setLoading(true);
        const result = await api.getIncidents({
          page,
          page_size: pageSize,
          status: status || undefined,
          store_id: store || undefined,
        });
        setIncidents(result.incidents);
        setTotal(result.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load incidents');
      } finally {
        setLoading(false);
      }
    }
    fetchIncidents();
  }, [page, status, store]);

  const updateParams = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams);
    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
    if (key !== 'page') {
      params.set('page', '1');
    }
    router.push(`/incidents?${params.toString()}`);
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Incidents</h1>
        <p className="text-gray-500 mt-1">
          Manage and investigate anomaly incidents
        </p>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="label">Status</label>
            <select
              className="input"
              value={status}
              onChange={(e) => updateParams('status', e.target.value)}
            >
              <option value="">All statuses</option>
              <option value="open">Open</option>
              <option value="investigating">Investigating</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="label">Store</label>
            <select
              className="input"
              value={store}
              onChange={(e) => updateParams('store', e.target.value)}
            >
              <option value="">All stores</option>
              {stores.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      ) : (
        <>
          <div className="card overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Incident
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Store
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    SKUs
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Detectors
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {incidents.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                      No incidents found
                    </td>
                  </tr>
                ) : (
                  incidents.map((incident) => {
                    const severity = getSeverityBadge(incident.severity_score);
                    const statusBadge = getStatusBadge(incident.status);

                    return (
                      <tr
                        key={incident.id}
                        className="hover:bg-gray-50 cursor-pointer"
                        onClick={() => router.push(`/incidents/${incident.id}`)}
                      >
                        <td className="px-4 py-4">
                          <p className="font-medium text-gray-900 truncate max-w-xs">
                            {incident.headline}
                          </p>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          {incident.store_id}
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          {incident.sku_count}
                        </td>
                        <td className="px-4 py-4">
                          <span className={clsx('badge', severity.class)}>
                            {severity.label}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <span className={clsx('badge', statusBadge.class)}>
                            {statusBadge.label}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          {format(parseISO(incident.date), 'dd MMM yyyy')}
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex gap-1">
                            {incident.detectors_triggered.map((detector) => (
                              <span
                                key={detector}
                                className="badge badge-info"
                              >
                                {detector === 'tukey' ? 'Tukey' : 'IF'}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                Showing {(page - 1) * pageSize + 1} to{' '}
                {Math.min(page * pageSize, total)} of {total} incidents
              </p>
              <div className="flex gap-2">
                <button
                  className="btn-secondary flex items-center gap-1"
                  disabled={page <= 1}
                  onClick={() => updateParams('page', String(page - 1))}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </button>
                <button
                  className="btn-secondary flex items-center gap-1"
                  disabled={page >= totalPages}
                  onClick={() => updateParams('page', String(page + 1))}
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
