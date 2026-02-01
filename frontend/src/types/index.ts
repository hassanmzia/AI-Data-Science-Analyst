// Dataset types
export interface Dataset {
  id: string;
  name: string;
  description: string;
  source: 'upload' | 'kaggle' | 'database' | 'url' | 'sample';
  file_format: string;
  row_count: number | null;
  column_count: number | null;
  columns_info: Record<string, any>;
  file_size: number | null;
  preview_data: any[];
  missing_values: Record<string, number>;
  dtypes: Record<string, string>;
  statistics: Record<string, any>;
  tags: string[];
  category: string;
  created_at: string;
  updated_at: string;
}

export interface DatabaseConnection {
  id: string;
  name: string;
  engine: string;
  host: string;
  port: number;
  database: string;
  username: string;
  is_active: boolean;
}

export interface Document {
  id: string;
  name: string;
  description: string;
  doc_type: string;
  file_size: number | null;
  is_indexed: boolean;
  chunk_count: number | null;
  created_at: string;
}

// Analysis types
export interface AnalysisSession {
  id: string;
  name: string;
  analysis_type: 'eda' | 'visualization' | 'hypothesis' | 'ml_model' | 'dl_model' | 'sql_query' | 'general';
  status: 'pending' | 'running' | 'completed' | 'failed';
  dataset: string | null;
  query: string;
  parameters: Record<string, any>;
  result: Record<string, any>;
  code_generated: string;
  error_message: string;
  execution_time: number | null;
  created_at: string;
}

export interface Visualization {
  id: string;
  name: string;
  chart_type: string;
  description: string;
  config: Record<string, any>;
  plotly_json: Record<string, any>;
  created_at: string;
}

export interface MLModel {
  id: string;
  name: string;
  model_type: string;
  task_type: string;
  framework: string;
  description: string;
  target_column: string;
  feature_columns: string[];
  metrics: Record<string, any>;
  epochs: number | null;
  batch_size: number | null;
  learning_rate: number | null;
  created_at: string;
}

export interface HypothesisTest {
  id: string;
  name: string;
  test_type: string;
  null_hypothesis: string;
  alt_hypothesis: string;
  test_statistic: number | null;
  p_value: number | null;
  reject_null: boolean | null;
  conclusion: string;
  created_at: string;
}

// Agent/Chat types
export interface Conversation {
  id: string;
  title: string;
  assistant_type: string;
  dataset: string | null;
  project: string | null;
  message_count: number;
  last_message: { role: string; content: string; created_at: string } | null;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  agent_name: string;
  tool_calls: any[];
  code_blocks: any[];
  visualizations: any[];
  metadata: Record<string, any>;
  created_at: string;
}

// Project types
export interface Project {
  id: string;
  name: string;
  description: string;
  project_type: string;
  status: 'active' | 'completed' | 'on_hold' | 'archived';
  tags: string[];
  notes: string;
  summary: string;
  key_findings: any[];
  dataset_count?: number;
  analysis_count?: number;
  created_at: string;
  updated_at: string;
}

// Archive types
export interface ArchivedProject {
  id: string;
  name: string;
  description: string;
  project_type: string;
  category: string;
  summary: string;
  key_findings: any[];
  methodology: string;
  tags: string[];
  tools_used: string[];
  is_template: boolean;
  use_count: number;
  analysis_count?: number;
  archived_at: string;
}

export interface ProjectTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  steps: any[];
  recommended_tools: string[];
  sample_queries: string[];
  use_count: number;
  rating: number;
  tags: string[];
}

// API Response
export interface PaginatedResponse<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}
