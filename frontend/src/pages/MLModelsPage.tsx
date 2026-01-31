import React, { useState, useEffect } from 'react';
import { BrainCircuit, Play, Loader2, Eye } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { analysisApi, datasetApi } from '../services/api';
import { Dataset, MLModel } from '../types';
import toast from 'react-hot-toast';

const MODEL_TYPES = [
  { value: 'auto', label: 'Auto Select' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'svm', label: 'SVM' },
  { value: 'neural_network', label: 'Neural Network' },
  { value: 'decision_tree', label: 'Decision Tree' },
  { value: 'knn', label: 'K-Nearest Neighbors' },
  { value: 'gradient_boosting', label: 'Gradient Boosting' },
  { value: 'linear_regression', label: 'Linear Regression' },
];

export default function MLModelsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<MLModel[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelType, setModelType] = useState('auto');
  const [targetColumn, setTargetColumn] = useState('');
  const [query, setQuery] = useState('');
  const [running, setRunning] = useState(false);
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
  const [columns, setColumns] = useState<string[]>([]);

  useEffect(() => { loadDatasets(); loadModels(); }, []);

  async function loadDatasets() {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadModels() {
    try {
      const res = await analysisApi.models.list();
      setModels(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function onDatasetChange(datasetId: string) {
    setSelectedDataset(datasetId);
    if (datasetId) {
      try {
        const res = await datasetApi.columnInfo(datasetId);
        setColumns(Object.keys(res.data));
      } catch { setColumns([]); }
    }
  }

  async function trainModel() {
    if (!selectedDataset || !query) {
      toast.error('Select dataset and describe the task');
      return;
    }
    setRunning(true);
    try {
      const res = await analysisApi.runMl({
        dataset_id: selectedDataset,
        query,
        model_type: modelType,
        target_column: targetColumn,
      });
      toast.success('Training started');

      if (res?.data?.id) {
        const poll = setInterval(async () => {
          try {
            const statusRes = await analysisApi.sessions.status(res.data.id);
            if (statusRes.data.status === 'completed' || statusRes.data.status === 'failed') {
              clearInterval(poll);
              loadModels();
              if (statusRes.data.status === 'completed') {
                toast.success('Model trained successfully');
              } else {
                toast.error('Training failed');
              }
            }
          } catch { clearInterval(poll); }
        }, 5000);
      }
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    } finally { setRunning(false); }
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">ML Models</h1>
      <p className="text-sm text-gray-500 mb-6">Train and evaluate machine learning models</p>

      {/* Training form */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <select value={selectedDataset} onChange={(e) => onDatasetChange(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Select Dataset</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            {MODEL_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
          <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Target Column (optional)</option>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button onClick={trainModel} disabled={running || !selectedDataset || !query}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 disabled:opacity-50 flex items-center justify-center">
            {running ? <Loader2 size={16} className="animate-spin mr-2" /> : <Play size={16} className="mr-2" />}
            Train Model
          </button>
        </div>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Train a logistic regression model to predict heart disease using all features"
          rows={2}
          className="w-full px-4 py-3 border rounded-lg text-sm resize-none"
        />
      </div>

      {/* Models list */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Trained Models</h2>
          {models.map(m => (
            <div key={m.id} onClick={() => setSelectedModel(m)}
              className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                selectedModel?.id === m.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <h3 className="font-medium text-gray-900 text-sm">{m.name}</h3>
              <p className="text-xs text-gray-500 mt-1">
                {m.model_type} | {m.task_type} | {new Date(m.created_at).toLocaleDateString()}
              </p>
              {m.metrics?.accuracy && (
                <p className="text-xs font-medium text-green-600 mt-1">
                  Accuracy: {(m.metrics.accuracy * 100).toFixed(1)}%
                </p>
              )}
            </div>
          ))}
          {models.length === 0 && <p className="text-center text-gray-500 py-6">No models trained yet</p>}
        </div>

        <div className="lg:col-span-2">
          {selectedModel ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{selectedModel.name}</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-500">Type</p>
                  <p className="text-sm font-medium">{selectedModel.model_type}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-500">Task</p>
                  <p className="text-sm font-medium">{selectedModel.task_type}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-500">Target</p>
                  <p className="text-sm font-medium">{selectedModel.target_column || 'N/A'}</p>
                </div>
              </div>
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{selectedModel.description}</ReactMarkdown>
              </div>
              {selectedModel.metrics && Object.keys(selectedModel.metrics).length > 0 && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-primary-600 font-medium">View Metrics</summary>
                  <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs overflow-auto">
                    {JSON.stringify(selectedModel.metrics, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <BrainCircuit size={40} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Select a model to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
