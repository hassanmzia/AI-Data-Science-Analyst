import React, { useState, useEffect, useCallback } from 'react';
import {
  Upload, Database, Globe, Download, Trash2, Eye, BarChart3,
  Plus, Search, FileSpreadsheet, RefreshCw, CheckCircle, XCircle, Loader2
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { datasetApi, connectionApi } from '../services/api';
import { Dataset } from '../types';
import toast from 'react-hot-toast';

interface DbConnection {
  id: string;
  name: string;
  engine: string;
  host: string;
  port: number;
  database: string;
  username: string;
  is_active: boolean;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [showKaggle, setShowKaggle] = useState(false);
  const [showUrl, setShowUrl] = useState(false);
  const [showDbImport, setShowDbImport] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [kaggleRef, setKaggleRef] = useState('');
  const [urlInput, setUrlInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Database import state
  const [connections, setConnections] = useState<DbConnection[]>([]);
  const [showNewConn, setShowNewConn] = useState(false);
  const [selectedConnId, setSelectedConnId] = useState('');
  const [dbTables, setDbTables] = useState<string[]>([]);
  const [dbQuery, setDbQuery] = useState('');
  const [dbImportName, setDbImportName] = useState('');
  const [dbLoading, setDbLoading] = useState(false);
  const [newConn, setNewConn] = useState({
    name: '', engine: 'postgresql', host: '', port: 5432,
    database: '', username: '', password: '',
  });

  useEffect(() => { loadDatasets(); }, []);

  async function loadDatasets() {
    setLoading(true);
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.results || res.data || []);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  }

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', file.name);
      try {
        await datasetApi.upload(formData);
        toast.success(`Uploaded ${file.name}`);
        loadDatasets();
      } catch (e: any) {
        toast.error(`Upload failed: ${e.response?.data?.error || e.message}`);
      }
    }
    setShowUpload(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'application/octet-stream': ['.parquet'],
    },
  });

  async function importKaggle() {
    if (!kaggleRef) return;
    try {
      await datasetApi.importKaggle({ dataset_ref: kaggleRef });
      toast.success('Kaggle import started');
      loadDatasets();
      setShowKaggle(false);
      setKaggleRef('');
    } catch (e: any) {
      toast.error(`Kaggle import failed: ${e.response?.data?.error || e.message}`);
    }
  }

  async function importUrl() {
    if (!urlInput) return;
    try {
      await datasetApi.importUrl({ url: urlInput });
      toast.success('URL import complete');
      loadDatasets();
      setShowUrl(false);
      setUrlInput('');
    } catch (e: any) {
      toast.error(`URL import failed: ${e.response?.data?.error || e.message}`);
    }
  }

  async function viewPreview(dataset: Dataset) {
    setSelectedDataset(dataset);
    try {
      const res = await datasetApi.preview(dataset.id);
      setPreview(res.data);
    } catch (e) { toast.error('Failed to load preview'); }
  }

  async function deleteDataset(id: string) {
    if (!window.confirm('Delete this dataset?')) return;
    try {
      await datasetApi.delete(id);
      toast.success('Deleted');
      loadDatasets();
      if (selectedDataset?.id === id) {
        setSelectedDataset(null);
        setPreview(null);
      }
    } catch (e) { toast.error('Delete failed'); }
  }

  async function loadConnections() {
    try {
      const res = await connectionApi.list();
      setConnections(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function createConnection() {
    try {
      await connectionApi.create(newConn);
      toast.success('Connection created');
      setShowNewConn(false);
      setNewConn({ name: '', engine: 'postgresql', host: '', port: 5432, database: '', username: '', password: '' });
      loadConnections();
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    }
  }

  async function testConnection(id: string) {
    try {
      const res = await connectionApi.test(id);
      if (res.data.success) toast.success('Connection successful');
      else toast.error(`Connection failed: ${res.data.message}`);
    } catch (e: any) {
      toast.error(`Connection failed: ${e.response?.data?.message || e.message}`);
    }
  }

  async function loadTables(connId: string) {
    setSelectedConnId(connId);
    setDbTables([]);
    try {
      const res = await connectionApi.tables(connId);
      setDbTables(res.data.tables || []);
    } catch (e: any) {
      toast.error(`Failed to list tables: ${e.response?.data?.error || e.message}`);
    }
  }

  async function importFromDb() {
    if (!selectedConnId || !dbQuery) {
      toast.error('Select a connection and enter a query or table name');
      return;
    }
    setDbLoading(true);
    try {
      await datasetApi.importDatabase({
        connection_id: selectedConnId,
        query: dbQuery,
        name: dbImportName || undefined,
      });
      toast.success('Database import complete');
      loadDatasets();
      setDbQuery('');
      setDbImportName('');
    } catch (e: any) {
      toast.error(`Import failed: ${e.response?.data?.error || e.message}`);
    } finally {
      setDbLoading(false);
    }
  }

  async function deleteConnection(id: string) {
    if (!window.confirm('Delete this connection?')) return;
    try {
      await connectionApi.delete(id);
      toast.success('Connection deleted');
      loadConnections();
      if (selectedConnId === id) { setSelectedConnId(''); setDbTables([]); }
    } catch (e) { toast.error('Delete failed'); }
  }

  useEffect(() => { if (showDbImport) loadConnections(); }, [showDbImport]);

  const filtered = datasets.filter(d =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-sm text-gray-500 mt-1">Upload, import, and manage your datasets</p>
        </div>
        <div className="flex space-x-2">
          <button onClick={() => setShowUpload(!showUpload)} className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 flex items-center">
            <Upload size={16} className="mr-2" /> Upload
          </button>
          <button onClick={() => setShowKaggle(!showKaggle)} className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm hover:bg-green-700 flex items-center">
            <Download size={16} className="mr-2" /> Kaggle
          </button>
          <button onClick={() => setShowUrl(!showUrl)} className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 flex items-center">
            <Globe size={16} className="mr-2" /> URL
          </button>
          <button onClick={() => setShowDbImport(!showDbImport)} className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm hover:bg-orange-700 flex items-center">
            <Database size={16} className="mr-2" /> Database
          </button>
        </div>
      </div>

      {/* Upload dropzone */}
      {showUpload && (
        <div className="mb-6 bg-white rounded-xl p-6 border-2 border-dashed border-gray-300">
          <div {...getRootProps()} className="text-center cursor-pointer py-8">
            <input {...getInputProps()} />
            <Upload size={40} className="mx-auto text-gray-400 mb-3" />
            {isDragActive ? (
              <p className="text-primary-600">Drop files here...</p>
            ) : (
              <div>
                <p className="text-gray-600">Drag & drop files here, or click to select</p>
                <p className="text-xs text-gray-400 mt-1">Supports CSV, Excel, JSON, Parquet</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Kaggle import */}
      {showKaggle && (
        <div className="mb-6 bg-white rounded-xl p-6 border border-gray-200">
          <h3 className="font-medium text-gray-900 mb-3">Import from Kaggle</h3>
          <div className="flex space-x-3">
            <input
              value={kaggleRef}
              onChange={(e) => setKaggleRef(e.target.value)}
              placeholder="e.g., uciml/heart-disease"
              className="flex-1 px-4 py-2 border rounded-lg text-sm"
            />
            <button onClick={importKaggle} className="px-6 py-2 bg-green-600 text-white rounded-lg text-sm">
              Import
            </button>
          </div>
        </div>
      )}

      {/* URL import */}
      {showUrl && (
        <div className="mb-6 bg-white rounded-xl p-6 border border-gray-200">
          <h3 className="font-medium text-gray-900 mb-3">Import from URL</h3>
          <div className="flex space-x-3">
            <input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://example.com/data.csv"
              className="flex-1 px-4 py-2 border rounded-lg text-sm"
            />
            <button onClick={importUrl} className="px-6 py-2 bg-purple-600 text-white rounded-lg text-sm">
              Import
            </button>
          </div>
        </div>
      )}

      {/* Database import */}
      {showDbImport && (
        <div className="mb-6 bg-white rounded-xl p-6 border border-gray-200 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900">Import from Database</h3>
            <button onClick={() => setShowNewConn(!showNewConn)}
              className="px-3 py-1 bg-orange-600 text-white rounded-lg text-xs hover:bg-orange-700 flex items-center">
              <Plus size={14} className="mr-1" /> New Connection
            </button>
          </div>

          {/* New connection form */}
          {showNewConn && (
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <input value={newConn.name} onChange={e => setNewConn({...newConn, name: e.target.value})}
                  placeholder="Connection name" className="px-3 py-2 border rounded-lg text-sm" />
                <select value={newConn.engine} onChange={e => setNewConn({...newConn, engine: e.target.value})}
                  className="px-3 py-2 border rounded-lg text-sm">
                  <option value="postgresql">PostgreSQL</option>
                  <option value="mysql">MySQL</option>
                  <option value="sqlite">SQLite</option>
                  <option value="mssql">MS SQL Server</option>
                </select>
                <input value={newConn.host} onChange={e => setNewConn({...newConn, host: e.target.value})}
                  placeholder="Host" className="px-3 py-2 border rounded-lg text-sm" />
                <input type="number" value={newConn.port} onChange={e => setNewConn({...newConn, port: parseInt(e.target.value) || 0})}
                  placeholder="Port" className="px-3 py-2 border rounded-lg text-sm" />
                <input value={newConn.database} onChange={e => setNewConn({...newConn, database: e.target.value})}
                  placeholder="Database name" className="px-3 py-2 border rounded-lg text-sm" />
                <input value={newConn.username} onChange={e => setNewConn({...newConn, username: e.target.value})}
                  placeholder="Username" className="px-3 py-2 border rounded-lg text-sm" />
                <input type="password" value={newConn.password} onChange={e => setNewConn({...newConn, password: e.target.value})}
                  placeholder="Password" className="px-3 py-2 border rounded-lg text-sm" />
              </div>
              <div className="flex space-x-2">
                <button onClick={createConnection} className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm">
                  Save Connection
                </button>
                <button onClick={() => setShowNewConn(false)} className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg text-sm">
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Existing connections */}
          {connections.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-gray-500 font-medium">Connections</p>
              {connections.map(conn => (
                <div key={conn.id} className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer ${
                  selectedConnId === conn.id ? 'border-orange-500 bg-orange-50' : 'border-gray-200 hover:bg-gray-50'
                }`} onClick={() => loadTables(conn.id)}>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{conn.name}</p>
                    <p className="text-xs text-gray-500">{conn.engine} | {conn.host}:{conn.port}/{conn.database}</p>
                  </div>
                  <div className="flex space-x-1">
                    <button onClick={(e) => { e.stopPropagation(); testConnection(conn.id); }}
                      className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200">
                      Test
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); deleteConnection(conn.id); }}
                      className="p-1 hover:bg-red-50 rounded">
                      <Trash2 size={14} className="text-gray-400 hover:text-red-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 text-center py-2">No connections yet. Create one to get started.</p>
          )}

          {/* Tables and query */}
          {selectedConnId && (
            <div className="space-y-3 border-t border-gray-200 pt-3">
              {dbTables.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 font-medium mb-1">Tables (click to use)</p>
                  <div className="flex flex-wrap gap-1">
                    {dbTables.map(t => (
                      <button key={t} onClick={() => setDbQuery(`SELECT * FROM ${t}`)}
                        className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-700">
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <input value={dbImportName} onChange={e => setDbImportName(e.target.value)}
                placeholder="Dataset name (optional)" className="w-full px-3 py-2 border rounded-lg text-sm" />
              <textarea value={dbQuery} onChange={e => setDbQuery(e.target.value)}
                placeholder="SQL query, e.g. SELECT * FROM my_table" rows={2}
                className="w-full px-3 py-2 border rounded-lg text-sm resize-none" />
              <button onClick={importFromDb} disabled={dbLoading || !dbQuery}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm hover:bg-orange-700 disabled:opacity-50 flex items-center">
                {dbLoading ? <Loader2 size={14} className="animate-spin mr-2" /> : <Database size={14} className="mr-2" />}
                Import Data
              </button>
            </div>
          )}
        </div>
      )}

      {/* Search */}
      <div className="mb-4">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search datasets..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg text-sm"
          />
        </div>
      </div>

      {/* Dataset list and preview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* List */}
        <div className="space-y-3">
          {filtered.map(d => (
            <div key={d.id} className={`bg-white rounded-xl p-4 border cursor-pointer hover:shadow-md transition-shadow ${
              selectedDataset?.id === d.id ? 'border-primary-500 ring-1 ring-primary-200' : 'border-gray-200'
            }`} onClick={() => viewPreview(d)}>
              <div className="flex items-start justify-between">
                <div className="flex items-center">
                  <FileSpreadsheet size={20} className="text-primary-500 mr-3" />
                  <div>
                    <h3 className="font-medium text-gray-900 text-sm">{d.name}</h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {d.source} | {d.row_count ? `${d.row_count.toLocaleString()} rows` : 'N/A'} | {d.column_count || 0} cols
                    </p>
                  </div>
                </div>
                <div className="flex space-x-1">
                  <button onClick={(e) => { e.stopPropagation(); deleteDataset(d.id); }}
                    className="p-1 hover:bg-red-50 rounded">
                    <Trash2 size={14} className="text-gray-400 hover:text-red-500" />
                  </button>
                </div>
              </div>
              {d.tags && d.tags.length > 0 && (
                <div className="flex gap-1 mt-2">
                  {d.tags.map(tag => (
                    <span key={tag} className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">{tag}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {filtered.length === 0 && !loading && (
            <p className="text-center text-gray-500 py-10">No datasets found. Upload one to get started.</p>
          )}
        </div>

        {/* Preview */}
        {preview && selectedDataset && (
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="p-4 border-b border-gray-200 bg-gray-50">
              <h3 className="font-medium text-gray-900">{selectedDataset.name}</h3>
              <p className="text-xs text-gray-500">
                {preview.total_rows} rows x {preview.columns?.length || 0} columns
              </p>
            </div>
            <div className="overflow-auto max-h-96">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    {preview.columns?.map((col: string) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-700 whitespace-nowrap">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.data?.slice(0, 50).map((row: any, i: number) => (
                    <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
                      {preview.columns?.map((col: string) => (
                        <td key={col} className="px-3 py-2 whitespace-nowrap text-gray-600">
                          {row[col] !== null ? String(row[col]).substring(0, 50) : '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
