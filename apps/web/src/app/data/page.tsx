'use client';

import { useEffect, useState, useRef } from 'react';
import {
  api,
  type Dataset,
  type AnalyzeCSVResponse,
  type ColumnAnalysis,
  type FlexibleDataStats,
} from '@/lib/api';
import {
  Upload,
  Database,
  Play,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Trash2,
  FileText,
  X,
  ChevronRight,
  AlertTriangle,
  Info,
  Plus,
  Settings,
  Eye,
} from 'lucide-react';
import { clsx } from 'clsx';
import { format, parseISO } from 'date-fns';

type WizardStep = 'select' | 'analyze' | 'configure' | 'upload';

interface SchemaConfig {
  name: string;
  description: string;
  dateColumn: string | null;
  identifierColumns: string[];
  metricColumns: string[];
  attributeColumns: string[];
}

export default function DataPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [stats, setStats] = useState<FlexibleDataStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'warning'; text: string } | null>(null);

  // Upload wizard state
  const [wizardOpen, setWizardOpen] = useState(false);
  const [wizardStep, setWizardStep] = useState<WizardStep>('select');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeCSVResponse | null>(null);
  const [schemaConfig, setSchemaConfig] = useState<SchemaConfig>({
    name: '',
    description: '',
    dateColumn: null,
    identifierColumns: [],
    metricColumns: [],
    attributeColumns: [],
  });
  const [wizardLoading, setWizardLoading] = useState(false);

  // Action states
  const [generating, setGenerating] = useState(false);
  const [runningDetection, setRunningDetection] = useState<number | null>(null);
  const [deletingDataset, setDeletingDataset] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchData = async () => {
    try {
      const [datasetsData, statsData] = await Promise.all([
        api.getDatasets(),
        api.getFlexibleDataStats(),
      ]);
      setDatasets(datasetsData.datasets);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to fetch data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // === Wizard Functions ===

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setWizardLoading(true);
    setWizardStep('analyze');

    try {
      const analysis = await api.analyzeCSV(file);
      setAnalysisResult(analysis);

      // Pre-populate schema config with suggestions
      setSchemaConfig({
        name: file.name.replace(/\.csv$/i, ''),
        description: '',
        dateColumn: analysis.suggested_date_column || null,
        identifierColumns: analysis.suggested_identifiers,
        metricColumns: analysis.suggested_metrics,
        attributeColumns: analysis.columns
          .filter(c => c.suggested_role === 'attribute')
          .map(c => c.name),
      });

      setWizardStep('configure');
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to analyze CSV',
      });
      setWizardStep('select');
    } finally {
      setWizardLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleToggleColumn = (column: string, role: 'identifier' | 'metric' | 'attribute') => {
    setSchemaConfig(prev => {
      const newConfig = { ...prev };

      // Remove from all arrays first
      newConfig.identifierColumns = newConfig.identifierColumns.filter(c => c !== column);
      newConfig.metricColumns = newConfig.metricColumns.filter(c => c !== column);
      newConfig.attributeColumns = newConfig.attributeColumns.filter(c => c !== column);

      // If it was already in that role, just remove it (toggle off)
      const wasInRole =
        (role === 'identifier' && prev.identifierColumns.includes(column)) ||
        (role === 'metric' && prev.metricColumns.includes(column)) ||
        (role === 'attribute' && prev.attributeColumns.includes(column));

      if (!wasInRole) {
        // Add to the new role
        if (role === 'identifier') {
          newConfig.identifierColumns = [...newConfig.identifierColumns, column];
        } else if (role === 'metric') {
          newConfig.metricColumns = [...newConfig.metricColumns, column];
        } else {
          newConfig.attributeColumns = [...newConfig.attributeColumns, column];
        }
      }

      return newConfig;
    });
  };

  const handleCreateAndUpload = async () => {
    if (!selectedFile || !schemaConfig.name) return;

    setWizardLoading(true);
    setWizardStep('upload');

    try {
      // Create dataset
      const dataset = await api.createDataset({
        name: schemaConfig.name,
        description: schemaConfig.description || undefined,
        date_column: schemaConfig.dateColumn || undefined,
        identifier_columns: schemaConfig.identifierColumns,
        metric_columns: schemaConfig.metricColumns,
        attribute_columns: schemaConfig.attributeColumns,
      });

      // Upload data
      const uploadResult = await api.uploadToDataset(dataset.id, selectedFile);

      setMessage({
        type: 'success',
        text: `Dataset "${dataset.name}" created with ${uploadResult.rows_inserted.toLocaleString()} rows.`,
      });

      closeWizard();
      await fetchData();
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to create dataset',
      });
      setWizardStep('configure');
    } finally {
      setWizardLoading(false);
    }
  };

  const closeWizard = () => {
    setWizardOpen(false);
    setWizardStep('select');
    setSelectedFile(null);
    setAnalysisResult(null);
    setSchemaConfig({
      name: '',
      description: '',
      dateColumn: null,
      identifierColumns: [],
      metricColumns: [],
      attributeColumns: [],
    });
  };

  // === Action Functions ===

  const handleGenerateDemo = async () => {
    setGenerating(true);
    setMessage(null);

    try {
      const result = await api.generateFlexibleDemoData('Demo Sales Data', 5, 50, 120);
      setMessage({
        type: 'success',
        text: `Demo dataset "${result.dataset_name}" created with ${result.row_count.toLocaleString()} rows. Metrics: ${result.metrics.join(', ')}`,
      });
      await fetchData();
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Generation failed',
      });
    } finally {
      setGenerating(false);
    }
  };

  const handleRunDetection = async (datasetId: number) => {
    setRunningDetection(datasetId);
    setMessage(null);

    try {
      const result = await api.runDatasetDetection(datasetId);

      if (result.error) {
        setMessage({
          type: 'error',
          text: result.error + (result.message ? `: ${result.message}` : ''),
        });
      } else {
        setMessage({
          type: 'success',
          text: `Detection completed on "${result.dataset_name}": ${result.total_outliers} outliers found, ${result.incidents_created} incidents created.`,
        });
      }
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Detection failed',
      });
    } finally {
      setRunningDetection(null);
    }
  };

  const handleDeleteDataset = async (datasetId: number, datasetName: string) => {
    if (!confirm(`Delete dataset "${datasetName}" and all its data?`)) return;

    setDeletingDataset(datasetId);
    setMessage(null);

    try {
      await api.deleteDataset(datasetId);
      setMessage({
        type: 'success',
        text: `Dataset "${datasetName}" deleted.`,
      });
      await fetchData();
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Delete failed',
      });
    } finally {
      setDeletingDataset(null);
    }
  };

  // === Helper Functions ===

  const getColumnRole = (column: string): string | null => {
    if (schemaConfig.dateColumn === column) return 'date';
    if (schemaConfig.identifierColumns.includes(column)) return 'identifier';
    if (schemaConfig.metricColumns.includes(column)) return 'metric';
    if (schemaConfig.attributeColumns.includes(column)) return 'attribute';
    return null;
  };

  const canProceed = schemaConfig.name.trim() !== '' && schemaConfig.metricColumns.length > 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  // === Upload Wizard Modal ===
  if (wizardOpen) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {wizardStep === 'select' && 'Upload New Dataset'}
              {wizardStep === 'analyze' && 'Analyzing CSV...'}
              {wizardStep === 'configure' && 'Configure Schema'}
              {wizardStep === 'upload' && 'Creating Dataset...'}
            </h1>
            {selectedFile && (
              <p className="text-gray-500 mt-1">
                {selectedFile.name}
                {analysisResult && ` • ${analysisResult.row_count.toLocaleString()} rows`}
              </p>
            )}
          </div>
          <button onClick={closeWizard} className="btn-secondary flex items-center gap-2">
            <X className="h-4 w-4" />
            Cancel
          </button>
        </div>

        {/* Step: Select File */}
        {wizardStep === 'select' && (
          <div className="card p-8 text-center">
            <Database className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Select a CSV File</h2>
            <p className="text-gray-500 mb-6">
              Upload any CSV file. We&apos;ll analyze the columns and help you configure the schema.
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              ref={fileInputRef}
              className="hidden"
              id="csv-wizard-upload"
            />
            <label
              htmlFor="csv-wizard-upload"
              className="btn-primary inline-flex items-center gap-2 cursor-pointer"
            >
              <Upload className="h-4 w-4" />
              Choose CSV File
            </label>
          </div>
        )}

        {/* Step: Analyzing */}
        {wizardStep === 'analyze' && (
          <div className="card p-8 text-center">
            <RefreshCw className="h-16 w-16 text-primary-500 mx-auto mb-4 animate-spin" />
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Analyzing your data...</h2>
            <p className="text-gray-500">Detecting column types and suggesting roles</p>
          </div>
        )}

        {/* Step: Configure Schema */}
        {wizardStep === 'configure' && analysisResult && (
          <div className="space-y-6">
            {/* Warnings */}
            {analysisResult.warnings.length > 0 && (
              <div className="space-y-2">
                {analysisResult.warnings.map((warning, idx) => (
                  <div key={idx} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-600 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-yellow-700">{warning}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Dataset Info */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Dataset Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={schemaConfig.name}
                    onChange={e => setSchemaConfig(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                    placeholder="My Sales Data"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <input
                    type="text"
                    value={schemaConfig.description}
                    onChange={e => setSchemaConfig(prev => ({ ...prev, description: e.target.value }))}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                    placeholder="Optional description"
                  />
                </div>
              </div>
            </div>

            {/* Date Column */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Date Column</h3>
              <p className="text-sm text-gray-500 mb-4">
                Select which column contains date/time values (optional but recommended for time-series analysis)
              </p>
              <select
                value={schemaConfig.dateColumn || ''}
                onChange={e => setSchemaConfig(prev => ({ ...prev, dateColumn: e.target.value || null }))}
                className="w-full md:w-1/2 rounded-lg border border-gray-300 px-3 py-2 text-sm"
              >
                <option value="">-- No date column --</option>
                {analysisResult.columns
                  .filter(c => ['date', 'datetime'].includes(c.detected_type) || c.suggested_role === 'date')
                  .map(col => (
                    <option key={col.name} value={col.name}>
                      {col.name} ({col.detected_type})
                    </option>
                  ))}
                <optgroup label="Other columns">
                  {analysisResult.columns
                    .filter(c => !['date', 'datetime'].includes(c.detected_type) && c.suggested_role !== 'date')
                    .map(col => (
                      <option key={col.name} value={col.name}>
                        {col.name} ({col.detected_type})
                      </option>
                    ))}
                </optgroup>
              </select>
            </div>

            {/* Column Configuration */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Column Roles</h3>
              <p className="text-sm text-gray-500 mb-4">
                Click on the role buttons to assign each column. <strong>Metrics</strong> (numeric columns to analyze) are required.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Column</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Sample Values</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {analysisResult.columns
                      .filter(col => col.name !== schemaConfig.dateColumn)
                      .map(col => {
                        const currentRole = getColumnRole(col.name);
                        const isNumeric = ['integer', 'float'].includes(col.detected_type);

                        return (
                          <tr key={col.name} className="hover:bg-gray-50">
                            <td className="px-3 py-3">
                              <span className="font-medium text-gray-900">{col.name}</span>
                              <span className="block text-xs text-gray-400">
                                {col.unique_count} unique / {col.total_count} total
                              </span>
                            </td>
                            <td className="px-3 py-3">
                              <span className={clsx(
                                'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
                                isNumeric ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
                              )}>
                                {col.detected_type}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-xs text-gray-500 font-mono max-w-xs truncate">
                              {col.sample_values.slice(0, 3).join(', ')}
                            </td>
                            <td className="px-3 py-3">
                              <div className="flex gap-1">
                                <button
                                  onClick={() => handleToggleColumn(col.name, 'identifier')}
                                  className={clsx(
                                    'px-2 py-1 rounded text-xs font-medium transition-colors',
                                    currentRole === 'identifier'
                                      ? 'bg-purple-600 text-white'
                                      : 'bg-gray-100 text-gray-600 hover:bg-purple-100'
                                  )}
                                >
                                  Identifier
                                </button>
                                <button
                                  onClick={() => handleToggleColumn(col.name, 'metric')}
                                  disabled={!isNumeric}
                                  className={clsx(
                                    'px-2 py-1 rounded text-xs font-medium transition-colors',
                                    currentRole === 'metric'
                                      ? 'bg-green-600 text-white'
                                      : isNumeric
                                      ? 'bg-gray-100 text-gray-600 hover:bg-green-100'
                                      : 'bg-gray-50 text-gray-300 cursor-not-allowed'
                                  )}
                                >
                                  Metric
                                </button>
                                <button
                                  onClick={() => handleToggleColumn(col.name, 'attribute')}
                                  className={clsx(
                                    'px-2 py-1 rounded text-xs font-medium transition-colors',
                                    currentRole === 'attribute'
                                      ? 'bg-orange-600 text-white'
                                      : 'bg-gray-100 text-gray-600 hover:bg-orange-100'
                                  )}
                                >
                                  Attribute
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>

              {/* Summary */}
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex flex-wrap gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Date:</span>{' '}
                    <span className="font-medium">{schemaConfig.dateColumn || 'None'}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Identifiers:</span>{' '}
                    <span className="font-medium text-purple-600">
                      {schemaConfig.identifierColumns.length > 0
                        ? schemaConfig.identifierColumns.join(', ')
                        : 'None'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Metrics:</span>{' '}
                    <span className="font-medium text-green-600">
                      {schemaConfig.metricColumns.length > 0
                        ? schemaConfig.metricColumns.join(', ')
                        : 'None (required)'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Attributes:</span>{' '}
                    <span className="font-medium text-orange-600">
                      {schemaConfig.attributeColumns.length > 0
                        ? schemaConfig.attributeColumns.join(', ')
                        : 'None'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Data Preview */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview</h3>
              <div className="overflow-x-auto max-h-64">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">#</th>
                      {analysisResult.columns.map(col => (
                        <th key={col.name} className="px-3 py-2 text-left text-xs font-medium text-gray-500">
                          {col.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {analysisResult.preview_rows.slice(0, 5).map((row, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-3 py-2 text-gray-400">{idx + 1}</td>
                        {analysisResult.columns.map(col => (
                          <td key={col.name} className="px-3 py-2 text-gray-700 font-mono text-xs">
                            {String(row[col.name] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-3">
              <button onClick={closeWizard} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={handleCreateAndUpload}
                disabled={!canProceed || wizardLoading}
                className={clsx(
                  'btn-primary flex items-center gap-2',
                  (!canProceed || wizardLoading) && 'opacity-50 cursor-not-allowed'
                )}
              >
                {wizardLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-4 w-4" />
                    Create Dataset &amp; Upload
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Step: Uploading */}
        {wizardStep === 'upload' && (
          <div className="card p-8 text-center">
            <RefreshCw className="h-16 w-16 text-primary-500 mx-auto mb-4 animate-spin" />
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Creating dataset...</h2>
            <p className="text-gray-500">Please wait while we upload your data</p>
          </div>
        )}
      </div>
    );
  }

  // === Main Data Page ===
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Data Management</h1>
          <p className="text-gray-500 mt-1">
            Upload datasets with any column structure and run anomaly detection
          </p>
        </div>
        <button
          onClick={() => setWizardOpen(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Dataset
        </button>
      </div>

      {/* Message */}
      {message && (
        <div
          className={clsx(
            'p-4 rounded-lg flex items-start gap-3',
            message.type === 'success'
              ? 'bg-green-50 border border-green-200'
              : message.type === 'warning'
              ? 'bg-yellow-50 border border-yellow-200'
              : 'bg-red-50 border border-red-200'
          )}
        >
          {message.type === 'success' ? (
            <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0" />
          ) : message.type === 'warning' ? (
            <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0" />
          ) : (
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
          )}
          <p
            className={clsx(
              'text-sm',
              message.type === 'success' ? 'text-green-700' : message.type === 'warning' ? 'text-yellow-700' : 'text-red-700'
            )}
          >
            {message.text}
          </p>
          <button
            onClick={() => setMessage(null)}
            className="ml-auto text-gray-400 hover:text-gray-600"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Stats Overview */}
      {stats && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Database className="h-5 w-5 text-gray-500" />
            Overview
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-500">Datasets</p>
              <p className="text-2xl font-bold text-gray-900">{stats.datasets.count}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-500">Total Rows</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.datasets.total_rows.toLocaleString()}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-500">Date Range</p>
              <p className="text-sm font-semibold text-gray-900">
                {stats.datasets.date_range.from && stats.datasets.date_range.to ? (
                  <>
                    {format(parseISO(stats.datasets.date_range.from), 'dd MMM yyyy')}
                    <br />
                    to {format(parseISO(stats.datasets.date_range.to), 'dd MMM yyyy')}
                  </>
                ) : (
                  'No data'
                )}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <button
                onClick={handleGenerateDemo}
                disabled={generating}
                className={clsx(
                  'btn-secondary w-full flex items-center justify-center gap-2',
                  generating && 'opacity-50 cursor-not-allowed'
                )}
              >
                {generating ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4" />
                )}
                Demo Data
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Datasets List */}
      <div className="card">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Datasets</h2>
        </div>

        {datasets.length === 0 ? (
          <div className="p-8 text-center">
            <Database className="h-12 w-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 mb-4">No datasets yet</p>
            <button
              onClick={() => setWizardOpen(true)}
              className="btn-primary inline-flex items-center gap-2"
            >
              <Upload className="h-4 w-4" />
              Upload Your First Dataset
            </button>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {datasets.map(dataset => (
              <div key={dataset.id} className="p-6 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-medium text-gray-900">{dataset.name}</h3>
                    {dataset.description && (
                      <p className="text-sm text-gray-500 mt-1">{dataset.description}</p>
                    )}
                    <div className="flex flex-wrap gap-4 mt-3 text-sm text-gray-500">
                      <span>{dataset.row_count.toLocaleString()} rows</span>
                      {dataset.date_range_start && dataset.date_range_end && (
                        <span>
                          {format(parseISO(dataset.date_range_start), 'dd MMM yyyy')} -{' '}
                          {format(parseISO(dataset.date_range_end), 'dd MMM yyyy')}
                        </span>
                      )}
                      <span>Created {format(parseISO(dataset.created_at), 'dd MMM yyyy')}</span>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-3">
                      {dataset.identifier_columns.map(col => (
                        <span
                          key={col}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700"
                        >
                          {col}
                        </span>
                      ))}
                      {dataset.metric_columns.map(col => (
                        <span
                          key={col}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700"
                        >
                          {col}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <button
                      onClick={() => handleRunDetection(dataset.id)}
                      disabled={runningDetection === dataset.id || dataset.metric_columns.length === 0}
                      className={clsx(
                        'btn-primary flex items-center gap-2',
                        (runningDetection === dataset.id || dataset.metric_columns.length === 0) &&
                          'opacity-50 cursor-not-allowed'
                      )}
                    >
                      {runningDetection === dataset.id ? (
                        <RefreshCw className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      Run Detection
                    </button>
                    <button
                      onClick={() => handleDeleteDataset(dataset.id, dataset.name)}
                      disabled={deletingDataset === dataset.id}
                      className={clsx(
                        'btn-secondary p-2 text-red-600 hover:bg-red-50',
                        deletingDataset === dataset.id && 'opacity-50 cursor-not-allowed'
                      )}
                      title="Delete dataset"
                    >
                      {deletingDataset === dataset.id ? (
                        <RefreshCw className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Help Section */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center font-bold">
              1
            </div>
            <div>
              <h4 className="font-medium text-gray-900">Upload Any CSV</h4>
              <p className="text-sm text-gray-500 mt-1">
                Upload your data file. We automatically detect column types (dates, numbers, categories).
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center font-bold">
              2
            </div>
            <div>
              <h4 className="font-medium text-gray-900">Configure Schema</h4>
              <p className="text-sm text-gray-500 mt-1">
                Choose which columns are identifiers (for grouping), metrics (to analyze), and attributes.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center font-bold">
              3
            </div>
            <div>
              <h4 className="font-medium text-gray-900">Run Detection</h4>
              <p className="text-sm text-gray-500 mt-1">
                Our algorithms (Tukey IQR + Isolation Forest) find anomalies in your metric columns.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
