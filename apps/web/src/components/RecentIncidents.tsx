import Link from 'next/link';
import { format, parseISO } from 'date-fns';
import { clsx } from 'clsx';
import type { Incident } from '@/lib/api';

interface RecentIncidentsProps {
  incidents: Incident[];
}

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

export function RecentIncidents({ incidents }: RecentIncidentsProps) {
  if (incidents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No recent incidents
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {incidents.map((incident) => {
        const severity = getSeverityBadge(incident.severity_score);
        const status = getStatusBadge(incident.status);

        return (
          <Link
            key={incident.id}
            href={`/incidents/${incident.id}`}
            className="block p-3 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50/50 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="font-medium text-gray-900 truncate">
                  {incident.headline}
                </p>
                <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
                  <span>{incident.store_id}</span>
                  <span>·</span>
                  <span>{incident.sku_count} SKU{incident.sku_count !== 1 ? 's' : ''}</span>
                  <span>·</span>
                  <span>{format(parseISO(incident.date), 'dd MMM yyyy')}</span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span className={clsx('badge', severity.class)}>
                  {severity.label}
                </span>
                <span className={clsx('badge', status.class)}>
                  {status.label}
                </span>
              </div>
            </div>
          </Link>
        );
      })}
      <Link
        href="/incidents"
        className="block text-center text-sm text-primary-600 hover:text-primary-700 font-medium pt-2"
      >
        View all incidents →
      </Link>
    </div>
  );
}
