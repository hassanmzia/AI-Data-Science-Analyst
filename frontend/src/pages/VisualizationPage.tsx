import React, { useState, useEffect } from 'react';
import { BarChart3, Play, Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import { analysisApi, datasetApi } from '../services/api';
import { Dataset, Visualization } from '../types';
import toast from 'react-hot-toast';

const CHART_TYPES = [
  { value: 'auto', label: 'Auto Detect' },
  { value: 'kde', label: 'KDE Plot' },
  { value: 'histogram', label: 'Histogram' },
  { value: 'scatter', label: 'Scatter Plot' },
  { value: 'bar', label: 'Bar Chart' },
  { value: 'line', label: 'Line Chart' },
  { value: 'heatmap', label: 'Heatmap' },
  { value: 'box', label: 'Box Plot' },
  { value: 'violin', label: 'Violin Plot' },
  { value: 'pie', label: 'Pie Chart' },
  { value: 'pair', label: 'Pair Plot' },
];

export default function VisualizationPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [visualizations, setVisualizations] = useState<Visualization[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [chartType, setChartType] = useState('auto');
  const [query, setQuery] = useState('');
  const [running, setRunning] = useState(false);
  const [selectedViz, setSelectedViz] = useState<Visualization | null>(null);

  useEffect(() => {
    loadDatasets();
    loadVisualizations();
  }, []);

  async function loadDatasets() {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadVisualizations() {
    try {
      const res = await analysisApi.visualizations.list();
      setVisualizations(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function createVisualization() {
    if (!selectedDataset || !query) {
      toast.error('Select a dataset and describe the visualization');
      return;
    }
    setRunning(true);
    try {
      const res = await analysisApi.runVisualization({
        dataset_id: selectedDataset,
        query,
        chart_type: chartType,
      });
      toast.success('Visualization started');

      // Poll for completion
      if (res?.data?.id) {
        const poll = setInterval(async () => {
          try {
            const statusRes = await analysisApi.sessions.status(res.data.id);
            if (statusRes.data.status === 'completed') {
              clearInterval(poll);
              loadVisualizations();
              if (statusRes.data.result?.visualization_id) {
                const vizRes = await analysisApi.visualizations.get(statusRes.data.result.visualization_id);
                setSelectedViz(vizRes.data);
              }
            } else if (statusRes.data.status === 'failed') {
              clearInterval(poll);
              toast.error('Visualization failed');
            }
          } catch { clearInterval(poll); }
        }, 3000);
      }
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    } finally {
      setRunning(false);
    }
  }

  function renderPlotly(plotlyJson: any) {
    if (!plotlyJson || !plotlyJson.data) return null;
    const layout = {
      ...(plotlyJson.layout || {}),
      autosize: true,
      margin: { t: 50, b: 50, l: 50, r: 30 },
    };
    return (
      <div className="bg-gray-50 rounded-lg p-4">
        <Plot
          data={plotlyJson.data}
          layout={layout}
          config={{ responsive: true, displayModeBar: true }}
          style={{ width: '100%', height: '400px' }}
          useResizeHandler
        />
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Visualization</h1>
      <p className="text-sm text-gray-500 mb-6">Create charts and plots using natural language</p>

      {/* Creation form */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Select Dataset</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <select value={chartType} onChange={(e) => setChartType(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            {CHART_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
          <button onClick={createVisualization} disabled={running || !selectedDataset || !query}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 disabled:opacity-50 flex items-center justify-center">
            {running ? <Loader2 size={16} className="animate-spin mr-2" /> : <Play size={16} className="mr-2" />}
            Create
          </button>
        </div>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Create a KDE plot comparing obesity distribution between CHD and non-CHD groups"
          rows={2}
          className="w-full px-4 py-3 border rounded-lg text-sm resize-none"
        />
      </div>

      {/* Visualization grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Gallery</h2>
          {visualizations.map(v => (
            <div key={v.id} onClick={() => setSelectedViz(v)}
              className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                selectedViz?.id === v.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <h3 className="font-medium text-gray-900 text-sm">{v.name}</h3>
              <p className="text-xs text-gray-500 mt-1">{v.chart_type} | {new Date(v.created_at).toLocaleDateString()}</p>
              {v.description && <p className="text-xs text-gray-400 mt-1">{v.description}</p>}
            </div>
          ))}
          {visualizations.length === 0 && (
            <p className="text-center text-gray-500 py-6">No visualizations yet</p>
          )}
        </div>

        <div>
          {selectedViz ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{selectedViz.name}</h3>
              <p className="text-sm text-gray-500 mb-4">{selectedViz.description}</p>
              {renderPlotly(selectedViz.plotly_json)}
              <details className="mt-4">
                <summary className="cursor-pointer text-xs text-primary-600">View Config</summary>
                <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs overflow-auto max-h-48">
                  {JSON.stringify(selectedViz.config, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <BarChart3 size={40} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Select a visualization to view</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
