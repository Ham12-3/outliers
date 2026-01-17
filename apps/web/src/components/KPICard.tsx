import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface KPICardProps {
  title: string;
  value: number | string;
  icon: LucideIcon;
  trend?: 'success' | 'warning' | 'danger' | 'neutral';
  subtitle?: string;
}

export function KPICard({ title, value, icon: Icon, trend = 'neutral', subtitle }: KPICardProps) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className={clsx(
            'text-2xl font-bold mt-1',
            trend === 'success' && 'text-green-600',
            trend === 'warning' && 'text-yellow-600',
            trend === 'danger' && 'text-red-600',
            trend === 'neutral' && 'text-gray-900'
          )}>
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
          )}
        </div>
        <div className={clsx(
          'p-3 rounded-lg',
          trend === 'success' && 'bg-green-100',
          trend === 'warning' && 'bg-yellow-100',
          trend === 'danger' && 'bg-red-100',
          trend === 'neutral' && 'bg-gray-100'
        )}>
          <Icon className={clsx(
            'h-6 w-6',
            trend === 'success' && 'text-green-600',
            trend === 'warning' && 'text-yellow-600',
            trend === 'danger' && 'text-red-600',
            trend === 'neutral' && 'text-gray-600'
          )} />
        </div>
      </div>
    </div>
  );
}
