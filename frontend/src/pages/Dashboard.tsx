import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database, MessageSquare, FlaskConical, BarChart3,
  BrainCircuit, FolderArchive, TrendingUp, Activity, Bot
} from 'lucide-react';
import { datasetApi, analysisApi, agentApi, projectApi, archiveApi } from '../services/api';

interface DashboardStats {
  datasets: number;
  analyses: number;
  conversations: number;
  projects: number;
  models: number;
  archives: number;
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    datasets: 0, analyses: 0, conversations: 0, projects: 0, models: 0, archives: 0,
  });
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      try {
        const [datasets, analyses, conversations, projects, models, archives] = await Promise.allSettled([
          datasetApi.list(),
          analysisApi.sessions.list(),
          agentApi.conversations.list(),
          projectApi.list(),
          analysisApi.models.list(),
          archiveApi.list(),
        ]);

        setStats({
          datasets: datasets.status === 'fulfilled' ? (datasets.value.data.results?.length || datasets.value.data.length || 0) : 0,
          analyses: analyses.status === 'fulfilled' ? (analyses.value.data.results?.length || analyses.value.data.length || 0) : 0,
          conversations: conversations.status === 'fulfilled' ? (conversations.value.data.results?.length || conversations.value.data.length || 0) : 0,
          projects: projects.status === 'fulfilled' ? (projects.value.data.results?.length || projects.value.data.length || 0) : 0,
          models: models.status === 'fulfilled' ? (models.value.data.results?.length || models.value.data.length || 0) : 0,
          archives: archives.status === 'fulfilled' ? (archives.value.data.results?.length || archives.value.data.length || 0) : 0,
        });

        if (analyses.status === 'fulfilled') {
          const results = analyses.value.data.results || analyses.value.data || [];
          setRecentAnalyses(results.slice(0, 5));
        }
      } catch (e) {
        console.error('Dashboard load error:', e);
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
  }, []);

  const statCards = [
    { label: 'Datasets', value: stats.datasets, icon: Database, color: 'bg-blue-500', link: '/datasets' },
    { label: 'Analyses', value: stats.analyses, icon: FlaskConical, color: 'bg-green-500', link: '/analysis' },
    { label: 'Conversations', value: stats.conversations, icon: MessageSquare, color: 'bg-purple-500', link: '/assistant' },
    { label: 'ML Models', value: stats.models, icon: BrainCircuit, color: 'bg-orange-500', link: '/ml-models' },
    { label: 'Projects', value: stats.projects, icon: FolderArchive, color: 'bg-indigo-500', link: '/projects' },
    { label: 'Archived', value: stats.archives, icon: FolderArchive, color: 'bg-gray-500', link: '/archive' },
  ];

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">AI Data Science Analyst</h1>
        <p className="mt-2 text-gray-600">
          Multi-agent AI system for data analysis, machine learning, and analytics
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {statCards.map((card) => (
          <Link
            key={card.label}
            to={card.link}
            className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">{card.label}</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{card.value}</p>
              </div>
              <div className={`${card.color} p-3 rounded-lg`}>
                <card.icon size={24} className="text-white" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            <Link to="/assistant" className="flex items-center p-3 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors">
              <Bot size={20} className="text-primary-600" />
              <span className="ml-2 text-sm font-medium text-primary-700">AI Assistant</span>
            </Link>
            <Link to="/datasets" className="flex items-center p-3 bg-green-50 rounded-lg hover:bg-green-100 transition-colors">
              <Database size={20} className="text-green-600" />
              <span className="ml-2 text-sm font-medium text-green-700">Upload Dataset</span>
            </Link>
            <Link to="/analysis" className="flex items-center p-3 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors">
              <FlaskConical size={20} className="text-purple-600" />
              <span className="ml-2 text-sm font-medium text-purple-700">Run Analysis</span>
            </Link>
            <Link to="/visualization" className="flex items-center p-3 bg-orange-50 rounded-lg hover:bg-orange-100 transition-colors">
              <BarChart3 size={20} className="text-orange-600" />
              <span className="ml-2 text-sm font-medium text-orange-700">Visualize</span>
            </Link>
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h2>
          {recentAnalyses.length > 0 ? (
            <div className="space-y-3">
              {recentAnalyses.map((a: any) => (
                <div key={a.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{a.name}</p>
                    <p className="text-xs text-gray-500">{a.analysis_type}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    a.status === 'completed' ? 'bg-green-100 text-green-700' :
                    a.status === 'running' ? 'bg-blue-100 text-blue-700' :
                    a.status === 'failed' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {a.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No analyses yet. Start by uploading a dataset!</p>
          )}
        </div>
      </div>

      {/* Capabilities */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Platform Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { title: 'Natural Language EDA', desc: 'Explore data using plain English queries' },
            { title: 'Auto Visualization', desc: 'Generate KDE plots, heatmaps, charts from text' },
            { title: 'ML Model Builder', desc: 'Train LogReg, RF, XGBoost, Neural Nets' },
            { title: 'Hypothesis Testing', desc: 'T-tests, chi-square, ANOVA from descriptions' },
            { title: 'SQL Querying', desc: 'Natural language to SQL translation' },
            { title: 'RAG Document Q&A', desc: 'Answer questions from uploaded documents' },
            { title: 'Multi-Agent System', desc: 'Specialized agents via MCP & A2A protocols' },
            { title: 'Kaggle Integration', desc: 'Import datasets directly from Kaggle' },
            { title: 'Project Archive', desc: 'Save and reuse past analyses as templates' },
          ].map((cap) => (
            <div key={cap.title} className="p-4 border border-gray-100 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-900">{cap.title}</h3>
              <p className="text-xs text-gray-500 mt-1">{cap.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
