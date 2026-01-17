const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Incident {
  id: number;
  date: string;
  store_id: string;
  status: 'open' | 'investigating' | 'resolved';
  severity_score: number;
  headline: string;
  description?: string;
  sku_count: number;
  estimated_impact?: number;
  detectors_triggered: string[];
  assignee?: string;
  resolution_reason?: string;
  resolved_at?: string;
  created_at: string;
  updated_at?: string;
  items?: IncidentItem[];
  notes?: IncidentNote[];
}

export interface IncidentItem {
  id: number;
  sku_id: string;
  detection_result_ids: number[];
  contribution_score?: number;
}

export interface IncidentNote {
  id: number;
  author: string;
  content: string;
  note_type: string;
  created_at: string;
}

export interface IncidentsSummary {
  status_counts: Record<string, number>;
  high_severity_count: number;
  affected_stores: number;
  affected_skus: number;
  total_estimated_impact: number;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
  is_outlier: boolean;
}

export interface TimeSeriesData {
  store_id: string;
  sku_id: string;
  metric: string;
  data: TimeSeriesPoint[];
}

export interface BoxplotData {
  metric: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
  iqr: number;
  lower_fence: number;
  upper_fence: number;
}

export interface DetectionStatus {
  running: boolean;
  last_run?: string;
  last_result?: {
    features_created?: number;
    tukey_outliers?: number;
    isolation_forest_outliers?: number;
    incidents_created?: number;
    incidents_updated?: number;
    error?: string;
  };
}

export interface DataStats {
  total_rows: number;
  store_count: number;
  sku_count: number;
  date_range: {
    from?: string;
    to?: string;
  };
}

export interface MappingCandidate {
  source_column: string;
  confidence: number;
  match_type: string;
}

export interface PreviewResponse {
  headers: string[];
  preview_rows: Record<string, unknown>[];
  row_count: number;
  suggested_mapping: Record<string, string | null>;
  mapping_candidates: Record<string, MappingCandidate[]>;
  unmapped_required: string[];
  unmapped_optional: string[];
  unmapped_source: string[];
  warnings: string[];
  errors: string[];
  can_proceed: boolean;
}

export interface UploadResponse {
  message: string;
  inserted: number;
  updated: number;
  skipped: number;
  errors: Array<{ row?: number; field?: string; error?: string }>;
  data_quality: {
    total_rows: number;
    valid_rows: number;
    error_rows: number;
    mapped_fields: string[];
    missing_fields: string[];
    has_on_hand: boolean;
    has_sold: boolean;
    detection_mode: string;
  };
}

export interface SchemaInfo {
  canonical_fields: Record<string, { required: boolean; type: string; default?: unknown }>;
  synonyms: Record<string, string[]>;
}

// Flexible dataset types
export interface ColumnAnalysis {
  name: string;
  detected_type: string;
  suggested_role: string;
  sample_values: unknown[];
  unique_count: number;
  null_count: number;
  total_count: number;
  min_value?: number;
  max_value?: number;
  mean_value?: number;
  std_value?: number;
  cardinality_ratio: number;
  role_confidence: number;
}

export interface AnalyzeCSVResponse {
  columns: ColumnAnalysis[];
  row_count: number;
  suggested_date_column?: string;
  suggested_identifiers: string[];
  suggested_metrics: string[];
  warnings: string[];
  preview_rows: Record<string, unknown>[];
}

export interface Dataset {
  id: number;
  name: string;
  description?: string;
  date_column?: string;
  identifier_columns: string[];
  metric_columns: string[];
  attribute_columns: string[];
  row_count: number;
  date_range_start?: string;
  date_range_end?: string;
  created_at: string;
  updated_at: string;
}

export interface DatasetSchemaRequest {
  name: string;
  description?: string;
  date_column?: string;
  identifier_columns: string[];
  metric_columns: string[];
  attribute_columns: string[];
}

export interface FlexibleUploadResponse {
  message: string;
  dataset_id: number;
  rows_inserted: number;
  rows_skipped: number;
  errors: Array<{ row?: number; error?: string }>;
}

export interface FlexibleDetectionResponse {
  dataset_id: number;
  dataset_name?: string;
  rows_analyzed: number;
  metrics_analyzed: string[];
  total_outliers: number;
  incidents_created: number;
  error?: string;
  message?: string;
}

export interface FlexibleDataStats {
  datasets: {
    count: number;
    total_rows: number;
    date_range: {
      from?: string;
      to?: string;
    };
  };
  legacy: {
    total_rows: number;
    store_count: number;
    sku_count: number;
    date_range: {
      from?: string;
      to?: string;
    };
  };
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API error: ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async health() {
    return this.fetch<{ status: string }>('/health');
  }

  // Incidents
  async getIncidents(params?: {
    status?: string;
    min_severity?: number;
    store_id?: string;
    date_from?: string;
    date_to?: string;
    page?: number;
    page_size?: number;
  }) {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
    }
    const query = searchParams.toString();
    return this.fetch<{
      incidents: Incident[];
      total: number;
      page: number;
      page_size: number;
    }>(`/incidents${query ? `?${query}` : ''}`);
  }

  async getIncidentsSummary() {
    return this.fetch<IncidentsSummary>('/incidents/summary');
  }

  async getIncident(id: number) {
    return this.fetch<Incident>(`/incidents/${id}`);
  }

  async updateIncident(id: number, data: {
    status?: string;
    assignee?: string;
    resolution_reason?: string;
  }) {
    return this.fetch<Incident>(`/incidents/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async addIncidentNote(id: number, content: string, author: string = 'user') {
    return this.fetch<IncidentNote>(`/incidents/${id}/notes`, {
      method: 'POST',
      body: JSON.stringify({ content, author }),
    });
  }

  async getIncidentDetectionDetails(id: number) {
    return this.fetch<{
      incident_id: number;
      date: string;
      store_id: string;
      detections_by_sku: Record<string, {
        tukey: Array<{
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
        }>;
        isolation_forest?: {
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
        };
      }>;
    }>(`/incidents/${id}/detection-details`);
  }

  // Metrics
  async getTimeseries(store_id: string, sku_id: string, date_from?: string, date_to?: string) {
    const params = new URLSearchParams({ store_id, sku_id });
    if (date_from) params.append('date_from', date_from);
    if (date_to) params.append('date_to', date_to);
    return this.fetch<TimeSeriesData[]>(`/metrics/timeseries?${params}`);
  }

  async getBoxplot(store_id: string, sku_id: string, date_from?: string, date_to?: string) {
    const params = new URLSearchParams({ store_id, sku_id });
    if (date_from) params.append('date_from', date_from);
    if (date_to) params.append('date_to', date_to);
    return this.fetch<BoxplotData[]>(`/metrics/boxplot?${params}`);
  }

  async getStores() {
    return this.fetch<{ stores: string[] }>('/metrics/stores');
  }

  async getSkus(store_id?: string) {
    const params = store_id ? `?store_id=${store_id}` : '';
    return this.fetch<{ skus: string[] }>(`/metrics/skus${params}`);
  }

  async getIncidentsOverTime(days: number = 30) {
    return this.fetch<{ data: { date: string; count: number }[] }>(
      `/metrics/incidents-over-time?days=${days}`
    );
  }

  async getSeverityDistribution() {
    return this.fetch<{ distribution: Record<string, number> }>(
      '/metrics/severity-distribution'
    );
  }

  async getTopStores(limit: number = 10) {
    return this.fetch<{ stores: { store_id: string; incident_count: number }[] }>(
      `/metrics/top-stores?limit=${limit}`
    );
  }

  // Data
  async getDataStats() {
    return this.fetch<DataStats>('/data/stats');
  }

  async getSchemaInfo() {
    return this.fetch<SchemaInfo>('/data/schema-info');
  }

  async previewCsv(file: File): Promise<PreviewResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/data/preview-csv`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Preview failed' }));
      throw new Error(error.detail || 'Preview failed');
    }

    return response.json();
  }

  async uploadCsv(file: File, mapping?: Record<string, string>): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (mapping) {
      formData.append('mapping', JSON.stringify(mapping));
    }

    const response = await fetch(`${this.baseUrl}/data/upload-csv`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      // Include more detail from the error response
      if (typeof error.detail === 'object') {
        throw new Error(error.detail.message || JSON.stringify(error.detail));
      }
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  async generateDemoData(stores: number = 5, skus: number = 50, days: number = 120) {
    return this.fetch<{
      message: string;
      stores: number;
      skus: number;
      days: number;
      total_rows: number;
    }>(`/data/generate-demo?stores=${stores}&skus=${skus}&days=${days}`, {
      method: 'POST',
    });
  }

  // Detection
  async runDetection(background: boolean = false) {
    return this.fetch<{
      message: string;
      status: string;
      features_created?: number;
      tukey_outliers?: number;
      isolation_forest_outliers?: number;
      incidents_created?: number;
      incidents_updated?: number;
    }>(`/detect/run?background=${background}`, {
      method: 'POST',
    });
  }

  async getDetectionStatus() {
    return this.fetch<DetectionStatus>('/detect/status');
  }

  // Flexible Dataset Endpoints
  async analyzeCSV(file: File): Promise<AnalyzeCSVResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/data/analyze-csv`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
  }

  async getDatasets() {
    return this.fetch<{ datasets: Dataset[]; total: number }>('/data/datasets');
  }

  async createDataset(schema: DatasetSchemaRequest) {
    return this.fetch<Dataset>('/data/datasets', {
      method: 'POST',
      body: JSON.stringify(schema),
    });
  }

  async getDataset(id: number) {
    return this.fetch<Dataset>(`/data/datasets/${id}`);
  }

  async deleteDataset(id: number) {
    return this.fetch<{ message: string }>(`/data/datasets/${id}`, {
      method: 'DELETE',
    });
  }

  async uploadToDataset(datasetId: number, file: File): Promise<FlexibleUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/data/datasets/${datasetId}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      if (typeof error.detail === 'object') {
        throw new Error(error.detail.message || JSON.stringify(error.detail));
      }
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  async getFlexibleDataStats() {
    return this.fetch<FlexibleDataStats>('/data/stats');
  }

  async generateFlexibleDemoData(
    datasetName: string = 'Demo Sales Data',
    stores: number = 5,
    products: number = 50,
    days: number = 120
  ) {
    const params = new URLSearchParams({
      dataset_name: datasetName,
      stores: String(stores),
      products: String(products),
      days: String(days),
    });
    return this.fetch<{
      message: string;
      dataset_id: number;
      dataset_name: string;
      row_count: number;
      identifiers: string[];
      metrics: string[];
    }>(`/data/generate-demo?${params}`, {
      method: 'POST',
    });
  }

  async runDatasetDetection(datasetId: number) {
    return this.fetch<FlexibleDetectionResponse>(`/detect/run/${datasetId}`, {
      method: 'POST',
    });
  }

  async getDatasetData(
    datasetId: number,
    params?: {
      limit?: number;
      offset?: number;
      identifier_filter?: string;
      date_from?: string;
      date_to?: string;
    }
  ) {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
    }
    const query = searchParams.toString();
    return this.fetch<{
      dataset_id: number;
      total: number;
      limit: number;
      offset: number;
      rows: Array<{
        id: number;
        date?: string;
        identifiers: Record<string, string>;
        metrics: Record<string, number>;
        attributes: Record<string, unknown>;
      }>;
    }>(`/data/datasets/${datasetId}/data${query ? `?${query}` : ''}`);
  }

  async getUniqueIdentifiers(datasetId: number) {
    return this.fetch<{
      dataset_id: number;
      identifier_columns: string[];
      unique_combinations: Array<{
        key: string;
        values: Record<string, string>;
      }>;
      count: number;
    }>(`/data/datasets/${datasetId}/unique-identifiers`);
  }
}

export const api = new ApiClient();
