'use client';

import { useEffect, useState } from 'react';
import { api, type Incident, type IncidentsSummary } from '@/lib/api';
import { KPICard } from '@/components/KPICard';
import { IncidentsOverTimeChart } from '@/components/charts/IncidentsOverTimeChart';
import { SeverityDistributionChart } from '@/components/charts/SeverityDistributionChart';
import { TopStoresChart } from '@/components/charts/TopStoresChart';
import { RecentIncidents } from '@/components/RecentIncidents';
import {
  AlertTriangle,
  Store,
  Package,
  TrendingUp,
} from 'lucide-react';

export default function DashboardPage() {
  const [summary, setSummary] = useState<IncidentsSummary | null>(null);
  const [recentIncidents, setRecentIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [summaryData, incidentsData] = await Promise.all([
          api.getIncidentsSummary(),
          api.getIncidents({ page_size: 5 }),
        ]);
        setSummary(summaryData);
        setRecentIncidents(incidentsData.incidents);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        {error}
      </div>
    );
  }

  const openCount = (summary?.status_counts?.open || 0) + (summary?.status_counts?.investigating || 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Overview of inventory anomalies and incidents
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Open Incidents"
          value={openCount}
          icon={AlertTriangle}
          trend={openCount > 0 ? 'warning' : 'success'}
        />
        <KPICard
          title="High Severity"
          value={summary?.high_severity_count || 0}
          icon={TrendingUp}
          trend={(summary?.high_severity_count || 0) > 0 ? 'danger' : 'success'}
        />
        <KPICard
          title="Affected Stores"
          value={summary?.affected_stores || 0}
          icon={Store}
        />
        <KPICard
          title="Affected SKUs"
          value={summary?.affected_skus || 0}
          icon={Package}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Incidents Over Time
          </h3>
          <IncidentsOverTimeChart />
        </div>
        <div className="card p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Severity Distribution
          </h3>
          <SeverityDistributionChart />
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Top Stores by Incidents
          </h3>
          <TopStoresChart />
        </div>
        <div className="card p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Recent Incidents
          </h3>
          <RecentIncidents incidents={recentIncidents} />
        </div>
      </div>
    </div>
  );
}
