import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://172.168.1.95:8050';
const MCP_URL = process.env.REACT_APP_MCP_URL || 'http://172.168.1.95:4050';

export const api = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
});

export const mcpApi = axios.create({
  baseURL: MCP_URL,
  headers: { 'Content-Type': 'application/json' },
});

// Dataset APIs
export const datasetApi = {
  list: () => api.get('/api/datasets/datasets/'),
  get: (id: string) => api.get(`/api/datasets/datasets/${id}/`),
  upload: (formData: FormData) =>
    api.post('/api/datasets/datasets/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  importKaggle: (data: { dataset_ref: string; name?: string }) =>
    api.post('/api/datasets/datasets/import_kaggle/', data),
  importUrl: (data: { url: string; name?: string }) =>
    api.post('/api/datasets/datasets/import_url/', data),
  importDatabase: (data: { connection_id: string; query: string; name?: string }) =>
    api.post('/api/datasets/datasets/import_database/', data),
  preview: (id: string) => api.get(`/api/datasets/datasets/${id}/preview/`),
  statistics: (id: string) => api.get(`/api/datasets/datasets/${id}/statistics/`),
  columnInfo: (id: string) => api.get(`/api/datasets/datasets/${id}/column_info/`),
  delete: (id: string) => api.delete(`/api/datasets/datasets/${id}/`),
  export: (id: string, format: string) =>
    api.post(`/api/datasets/datasets/${id}/export/`, { format }),
};

// Database Connection APIs
export const connectionApi = {
  list: () => api.get('/api/datasets/connections/'),
  create: (data: any) => api.post('/api/datasets/connections/', data),
  test: (id: string) => api.post(`/api/datasets/connections/${id}/test_connection/`),
  tables: (id: string) => api.get(`/api/datasets/connections/${id}/tables/`),
  delete: (id: string) => api.delete(`/api/datasets/connections/${id}/`),
};

// Document APIs
export const documentApi = {
  list: () => api.get('/api/datasets/documents/'),
  upload: (formData: FormData) =>
    api.post('/api/datasets/documents/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  query: (id: string, question: string) =>
    api.post(`/api/datasets/documents/${id}/query/`, { question }),
  delete: (id: string) => api.delete(`/api/datasets/documents/${id}/`),
};

// Analysis APIs
export const analysisApi = {
  sessions: {
    list: () => api.get('/api/analysis/sessions/'),
    get: (id: string) => api.get(`/api/analysis/sessions/${id}/`),
    status: (id: string) => api.get(`/api/analysis/sessions/${id}/status_check/`),
  },
  runEda: (data: { dataset_id: string; query: string; name?: string }) =>
    api.post('/api/analysis/sessions/run_eda/', data),
  runVisualization: (data: { dataset_id: string; query: string; chart_type?: string }) =>
    api.post('/api/analysis/sessions/run_visualization/', data),
  runMl: (data: { dataset_id: string; query: string; model_type?: string; target_column?: string }) =>
    api.post('/api/analysis/sessions/run_ml/', data),
  runHypothesisTest: (data: { dataset_id: string; query: string; test_type?: string }) =>
    api.post('/api/analysis/sessions/run_hypothesis_test/', data),
  runSql: (data: { dataset_id?: string; connection_id?: string; query: string }) =>
    api.post('/api/analysis/sessions/run_sql/', data),
  visualizations: {
    list: () => api.get('/api/analysis/visualizations/'),
    get: (id: string) => api.get(`/api/analysis/visualizations/${id}/`),
  },
  models: {
    list: () => api.get('/api/analysis/models/'),
    get: (id: string) => api.get(`/api/analysis/models/${id}/`),
    predict: (id: string, data: any) =>
      api.post(`/api/analysis/models/${id}/predict/`, data),
  },
  hypothesisTests: {
    list: () => api.get('/api/analysis/hypothesis-tests/'),
  },
};

// Conversation/Agent APIs
export const agentApi = {
  conversations: {
    list: () => api.get('/api/agents/conversations/'),
    create: (data: { title: string; assistant_type: string; dataset?: string }) =>
      api.post('/api/agents/conversations/', data),
    get: (id: string) => api.get(`/api/agents/conversations/${id}/`),
    chat: (id: string, data: { message: string; dataset_id?: string; document_id?: string }) =>
      api.post(`/api/agents/conversations/${id}/chat/`, data),
    messages: (id: string) => api.get(`/api/agents/conversations/${id}/messages/`),
    archive: (id: string) => api.post(`/api/agents/conversations/${id}/archive/`),
    delete: (id: string) => api.delete(`/api/agents/conversations/${id}/`),
  },
  configs: {
    list: () => api.get('/api/agents/configs/'),
  },
};

// Project APIs
export const projectApi = {
  list: () => api.get('/api/projects/projects/'),
  create: (data: { name: string; description: string; project_type: string; tags?: string[] }) =>
    api.post('/api/projects/projects/', data),
  get: (id: string) => api.get(`/api/projects/projects/${id}/`),
  update: (id: string, data: any) => api.patch(`/api/projects/projects/${id}/`, data),
  delete: (id: string) => api.delete(`/api/projects/projects/${id}/`),
  generateSummary: (id: string) => api.post(`/api/projects/projects/${id}/generate_summary/`),
  archive: (id: string) => api.post(`/api/projects/projects/${id}/archive/`),
};

// Archive APIs
export const archiveApi = {
  list: () => api.get('/api/archive/projects/'),
  get: (id: string) => api.get(`/api/archive/projects/${id}/`),
  search: (data: { query: string; category?: string; tags?: string[] }) =>
    api.post('/api/archive/projects/search/', data),
  createFromArchive: (id: string, data: { name: string; description?: string }) =>
    api.post(`/api/archive/projects/${id}/create_project_from_archive/`, data),
  makeTemplate: (id: string) => api.post(`/api/archive/projects/${id}/make_template/`),
  similar: (id: string) => api.get(`/api/archive/projects/${id}/similar/`),
  templates: {
    list: () => api.get('/api/archive/templates/'),
    useTemplate: (id: string, data: { name: string }) =>
      api.post(`/api/archive/templates/${id}/use_template/`, data),
  },
};

// MCP APIs
export const mcpToolsApi = {
  listTools: () => mcpApi.post('/mcp/tools/list'),
  callTool: (name: string, args: any) => mcpApi.post('/mcp/tools/call', { name, arguments: args }),
  listResources: () => mcpApi.post('/mcp/resources/list'),
  listAgents: () => mcpApi.get('/a2a/agents'),
  agentChat: (data: any) => mcpApi.post('/api/agent/chat', data),
};
