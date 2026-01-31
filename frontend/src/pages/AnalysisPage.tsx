import React, { useState, useEffect } from 'react';
import { FlaskConical, Play, Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { analysisApi, datasetApi } from '../services/api';
import { Dataset, AnalysisSession } from '../types';
import toast from 'react-hot-toast';

const ANALYSIS_TYPES = [
  { value: 'eda', label: 'Exploratory Data Analysis', placeholder: 'e.g., What are the different columns? How many missing values?' },
  { value: 'hypothesis', label: 'Hypothesis Testing', placeholder: 'e.g., Test if higher obesity increases CHD risk' },
  { value: 'sql', label: 'SQL Query', placeholder: 'e.g., What is the maximum salary?' },
];

export default function AnalysisPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [sessions, setSessions] = useState<AnalysisSession[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [analysisType, setAnalysisType] = useState('eda');
  const [query, setQuery] = useState('');
  const [running, setRunning] = useState(false);
  const [selectedSession, setSelectedSession] = useState<AnalysisSession | null>(null);

  useEffect(() => {
    loadDatasets();
    loadSessions();
  }, []);

  async function loadDatasets() {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadSessions() {
    try {
      const res = await analysisApi.sessions.list();
      setSessions(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function runAnalysis() {
    if (!selectedDataset || !query) {
      toast.error('Select a dataset and enter a query');
      return;
    }
    setRunning(true);
    try {
      let res;
      if (analysisType === 'eda') {
        res = await analysisApi.runEda({ dataset_id: selectedDataset, query });
      } else if (analysisType === 'hypothesis') {
        res = await analysisApi.runHypothesisTest({ dataset_id: selectedDataset, query });
      } else if (analysisType === 'sql') {
        res = await analysisApi.runSql({ dataset_id: selectedDataset, query });
      }

      toast.success('Analysis started');
      loadSessions();

      // Poll for results
      if (res?.data?.id) {
        pollStatus(res.data.id);
      }
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    } finally {
      setRunning(false);
    }
  }

  async function pollStatus(sessionId: string) {
    const poll = setInterval(async () => {
      try {
        const res = await analysisApi.sessions.status(sessionId);
        if (res.data.status === 'completed' || res.data.status === 'failed') {
          clearInterval(poll);
          loadSessions();
          const full = await analysisApi.sessions.get(sessionId);
          setSelectedSession(full.data);
        }
      } catch (e) { clearInterval(poll); }
    }, 3000);
  }

  async function viewSession(session: AnalysisSession) {
    try {
      const res = await analysisApi.sessions.get(session.id);
      setSelectedSession(res.data);
    } catch (e) { toast.error('Failed to load'); }
  }

  const statusIcon = (s: string) => {
    switch (s) {
      case 'completed': return <CheckCircle size={16} className="text-green-500" />;
      case 'failed': return <XCircle size={16} className="text-red-500" />;
      case 'running': return <Loader2 size={16} className="text-blue-500 animate-spin" />;
      default: return <Clock size={16} className="text-gray-400" />;
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Analysis</h1>
      <p className="text-sm text-gray-500 mb-6">Run EDA, hypothesis tests, and SQL queries using natural language</p>

      {/* Analysis Form */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            <option value="">Select Dataset</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <select value={analysisType} onChange={(e) => setAnalysisType(e.target.value)}
            className="px-4 py-2 border rounded-lg text-sm">
            {ANALYSIS_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
          <button onClick={runAnalysis} disabled={running || !selectedDataset || !query}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 disabled:opacity-50 flex items-center justify-center">
            {running ? <Loader2 size={16} className="animate-spin mr-2" /> : <Play size={16} className="mr-2" />}
            Run Analysis
          </button>
        </div>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={ANALYSIS_TYPES.find(t => t.value === analysisType)?.placeholder}
          rows={3}
          className="w-full px-4 py-3 border rounded-lg text-sm resize-none"
        />
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Session list */}
        <div className="space-y-2">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">History</h2>
          {sessions.map(s => (
            <div key={s.id} onClick={() => viewSession(s)}
              className={`bg-white rounded-lg p-3 border cursor-pointer hover:shadow-sm ${
                selectedSession?.id === s.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 truncate">{s.name}</span>
                {statusIcon(s.status)}
              </div>
              <p className="text-xs text-gray-500 mt-1">{s.analysis_type} | {new Date(s.created_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>

        {/* Session detail */}
        <div className="lg:col-span-2">
          {selectedSession ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">{selectedSession.name}</h3>
                <div className="flex items-center space-x-2">
                  {statusIcon(selectedSession.status)}
                  <span className="text-sm text-gray-500">{selectedSession.status}</span>
                </div>
              </div>
              {selectedSession.execution_time && (
                <p className="text-xs text-gray-400 mb-4">Execution time: {selectedSession.execution_time.toFixed(2)}s</p>
              )}
              <div className="prose prose-sm max-w-none">
                {selectedSession.result?.answer && (
                  <ReactMarkdown>{selectedSession.result.answer}</ReactMarkdown>
                )}
                {selectedSession.result?.conclusion && (
                  <ReactMarkdown>{selectedSession.result.conclusion}</ReactMarkdown>
                )}
                {selectedSession.result?.summary && (
                  <ReactMarkdown>{selectedSession.result.summary}</ReactMarkdown>
                )}
                {selectedSession.error_message && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4">
                    <p className="text-red-700 text-sm">{selectedSession.error_message}</p>
                  </div>
                )}
                {selectedSession.result?.describe && (
                  <details className="mt-4">
                    <summary className="cursor-pointer text-sm text-primary-600 font-medium">View Statistics</summary>
                    <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs overflow-auto">
                      {JSON.stringify(selectedSession.result.describe, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <FlaskConical size={40} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Select an analysis to view results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
