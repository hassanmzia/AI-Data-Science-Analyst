import React, { useState, useEffect } from 'react';
import { BrainCircuit, Play, Loader2, Cpu, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { analysisApi, datasetApi } from '../services/api';
import { Dataset, MLModel } from '../types';
import toast from 'react-hot-toast';

const ML_MODEL_TYPES = [
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

const DL_MODEL_TYPES = [
  { value: 'auto', label: 'Auto Select' },
  { value: 'mlp', label: 'Multi-Layer Perceptron (MLP)' },
  { value: 'cnn', label: 'CNN (Convolutional)' },
  { value: 'rnn', label: 'RNN (Recurrent)' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'gru', label: 'GRU' },
  { value: 'transformer', label: 'Transformer' },
  { value: 'autoencoder', label: 'Autoencoder' },
  { value: 'gan', label: 'GAN (Generative)' },
  { value: 'resnet', label: 'ResNet' },
];

const DL_FRAMEWORKS = [
  { value: 'pytorch', label: 'PyTorch' },
  { value: 'tensorflow', label: 'TensorFlow / Keras' },
  { value: 'auto', label: 'Auto Select' },
];

const DL_TASK_TYPES = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'sequence_prediction', label: 'Sequence Prediction' },
  { value: 'anomaly_detection', label: 'Anomaly Detection' },
  { value: 'generative', label: 'Generative' },
];

type Tab = 'ml' | 'dl';

export default function MLModelsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<MLModel[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelType, setModelType] = useState('auto');
  const [targetColumn, setTargetColumn] = useState('');
  const [query, setQuery] = useState('');
  const [running, setRunning] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>('ml');

  // DL-specific state
  const [dlFramework, setDlFramework] = useState('pytorch');
  const [dlTaskType, setDlTaskType] = useState('auto');
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);

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

  function onTabChange(tab: Tab) {
    setActiveTab(tab);
    setModelType('auto');
    setQuery('');
  }

  async function trainModel() {
    if (!selectedDataset || !query) {
      toast.error('Select dataset and describe the task');
      return;
    }
    setRunning(true);
    try {
      let res;
      if (activeTab === 'dl') {
        res = await analysisApi.runDl({
          dataset_id: selectedDataset,
          query,
          model_type: modelType,
          framework: dlFramework,
          target_column: targetColumn,
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          task_type: dlTaskType,
        });
        toast.success('Deep learning training started');
      } else {
        res = await analysisApi.runMl({
          dataset_id: selectedDataset,
          query,
          model_type: modelType,
          target_column: targetColumn,
        });
        toast.success('ML training started');
      }

      if (res?.data?.id) {
        setTrainingStatus('Training in progress...');
        const poll = setInterval(async () => {
          try {
            const statusRes = await analysisApi.sessions.status(res.data.id);
            if (statusRes.data.status === 'running') {
              setTrainingStatus('Training in progress...');
            }
            if (statusRes.data.status === 'completed' || statusRes.data.status === 'failed') {
              clearInterval(poll);
              setTrainingStatus('');
              if (statusRes.data.status === 'completed') {
                toast.success('Model trained successfully');
                const modelsRes = await analysisApi.models.list();
                const allModels = modelsRes.data.results || modelsRes.data || [];
                setModels(allModels);
                // Auto-select the newest model
                if (allModels.length > 0) {
                  setSelectedModel(allModels[0]);
                }
              } else {
                toast.error(`Training failed: ${statusRes.data.error_message || 'Unknown error'}`);
                loadModels();
              }
            }
          } catch { clearInterval(poll); setTrainingStatus(''); }
        }, 5000);
      }
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    } finally { setRunning(false); }
  }

  const isDL = activeTab === 'dl';
  const currentModelTypes = isDL ? DL_MODEL_TYPES : ML_MODEL_TYPES;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">ML & Deep Learning Models</h1>
      <p className="text-sm text-gray-500 mb-6">Train and evaluate machine learning and deep learning models</p>

      {/* Tabs */}
      <div className="flex space-x-1 bg-gray-100 rounded-lg p-1 mb-6 max-w-md">
        <button
          onClick={() => onTabChange('ml')}
          className={`flex-1 flex items-center justify-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'ml'
              ? 'bg-white text-primary-700 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Cpu size={16} className="mr-2" />
          Machine Learning
        </button>
        <button
          onClick={() => onTabChange('dl')}
          className={`flex-1 flex items-center justify-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'dl'
              ? 'bg-white text-primary-700 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Zap size={16} className="mr-2" />
          Deep Learning
        </button>
      </div>

      {/* Training form */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 mb-6">
        <h2 className="text-sm font-semibold text-gray-700 mb-4">
          {isDL ? 'Deep Learning Training' : 'Machine Learning Training'}
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <select value={selectedDataset} onChange={(e) => onDatasetChange(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Select Dataset</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            {currentModelTypes.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
          <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Target Column (optional)</option>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button onClick={trainModel} disabled={running || !selectedDataset || !query}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 disabled:opacity-50 flex items-center justify-center">
            {running ? <Loader2 size={16} className="animate-spin mr-2" /> : <Play size={16} className="mr-2" />}
            {isDL ? 'Train DL Model' : 'Train Model'}
          </button>
        </div>

        {/* DL-specific options */}
        {isDL && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Framework</label>
              <select value={dlFramework} onChange={(e) => setDlFramework(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg text-sm">
                {DL_FRAMEWORKS.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Task Type</label>
              <select value={dlTaskType} onChange={(e) => setDlTaskType(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg text-sm">
                {DL_TASK_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Epochs</label>
              <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))}
                min={1} max={1000}
                className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Batch Size</label>
              <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))}
                min={1} max={4096}
                className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Learning Rate</label>
              <input type="number" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))}
                step={0.0001} min={0.00001} max={1}
                className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
          </div>
        )}

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={isDL
            ? 'e.g., Train an LSTM model to predict stock prices using the last 10 time steps'
            : 'e.g., Train a logistic regression model to predict heart disease using all features'
          }
          rows={2}
          className="w-full px-4 py-3 border rounded-lg text-sm resize-none"
        />
      </div>

      {/* Training status */}
      {trainingStatus && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6 flex items-center">
          <Loader2 size={18} className="animate-spin text-blue-600 mr-3" />
          <span className="text-sm text-blue-700 font-medium">{trainingStatus}</span>
        </div>
      )}

      {/* Models list */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Trained Models</h2>
          {models.map(m => (
            <div key={m.id} onClick={() => setSelectedModel(m)}
              className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                selectedModel?.id === m.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-gray-900 text-sm">{m.name}</h3>
                {isDLModel(m.model_type) && (
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">DL</span>
                )}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {m.model_type} | {m.task_type}
                {m.framework && m.framework !== 'sklearn' ? ` | ${m.framework}` : ''}
                {' | '}{new Date(m.created_at).toLocaleDateString()}
              </p>
              {m.metrics?.accuracy ? (
                <p className="text-xs font-medium text-green-600 mt-1">
                  Accuracy: {(m.metrics.accuracy * (m.metrics.accuracy <= 1 ? 100 : 1)).toFixed(1)}%
                </p>
              ) : m.metrics?.summary ? (
                <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                  {typeof m.metrics.summary === 'string'
                    ? m.metrics.summary.replace(/[*#\n]/g, ' ').substring(0, 80) + '...'
                    : ''}
                </p>
              ) : null}
              {m.epochs && (
                <p className="text-xs text-gray-400 mt-1">
                  {m.epochs} epochs
                </p>
              )}
            </div>
          ))}
          {models.length === 0 && <p className="text-center text-gray-500 py-6">No models trained yet</p>}
        </div>

        <div className="lg:col-span-2">
          {selectedModel ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900">{selectedModel.name}</h3>
                {isDLModel(selectedModel.model_type) && (
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full font-medium">
                    Deep Learning
                  </span>
                )}
              </div>
              <div className={`grid gap-4 mb-4 ${isDLModel(selectedModel.model_type) ? 'grid-cols-3 md:grid-cols-6' : 'grid-cols-3'}`}>
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
                {isDLModel(selectedModel.model_type) && (
                  <>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="text-xs text-gray-500">Framework</p>
                      <p className="text-sm font-medium">{selectedModel.framework || 'N/A'}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="text-xs text-gray-500">Epochs</p>
                      <p className="text-sm font-medium">{selectedModel.epochs ?? 'N/A'}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="text-xs text-gray-500">Learning Rate</p>
                      <p className="text-sm font-medium">{selectedModel.learning_rate ?? 'N/A'}</p>
                    </div>
                  </>
                )}
              </div>
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{getModelSummary(selectedModel)}</ReactMarkdown>
              </div>
              {selectedModel.metrics && Object.keys(selectedModel.metrics).length > 0 && (
                <div className="mt-4">
                  {renderMetrics(selectedModel.metrics)}
                </div>
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

const DL_TYPES = new Set(['cnn', 'rnn', 'lstm', 'gru', 'transformer', 'autoencoder', 'gan', 'mlp', 'resnet']);

function isDLModel(modelType: string): boolean {
  return DL_TYPES.has(modelType);
}

function getModelSummary(model: MLModel): string {
  if (model.description) return model.description;
  if (model.metrics?.summary && typeof model.metrics.summary === 'string') return model.metrics.summary;
  return 'No description available.';
}

function renderMetrics(metrics: Record<string, any>) {
  // If metrics only contains a 'summary' string, don't show a separate metrics section
  // since it's already rendered as the description above
  const keys = Object.keys(metrics).filter(k => k !== 'summary');

  // Extract numeric metrics from the summary text or from top-level keys
  const numericMetrics: { label: string; value: string }[] = [];
  for (const key of keys) {
    const val = metrics[key];
    if (typeof val === 'number') {
      const isPercent = key.toLowerCase().includes('accuracy') || key.toLowerCase().includes('precision')
        || key.toLowerCase().includes('recall') || key.toLowerCase().includes('f1');
      numericMetrics.push({
        label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        value: isPercent ? `${(val * (val <= 1 ? 100 : 1)).toFixed(2)}%` : val.toFixed(4),
      });
    } else if (typeof val === 'string' && val.length < 100) {
      numericMetrics.push({
        label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        value: val,
      });
    }
  }

  if (numericMetrics.length === 0 && keys.length === 0) return null;

  if (numericMetrics.length > 0) {
    return (
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {numericMetrics.map(m => (
            <div key={m.label} className="bg-green-50 border border-green-100 p-3 rounded-lg text-center">
              <p className="text-xs text-gray-500">{m.label}</p>
              <p className="text-lg font-bold text-green-700">{m.value}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Fallback: show non-summary keys as formatted JSON
  const filtered = Object.fromEntries(Object.entries(metrics).filter(([k]) => k !== 'summary'));
  if (Object.keys(filtered).length === 0) return null;
  return (
    <details>
      <summary className="cursor-pointer text-sm text-primary-600 font-medium">View Raw Metrics</summary>
      <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs overflow-auto">
        {JSON.stringify(filtered, null, 2)}
      </pre>
    </details>
  );
}
