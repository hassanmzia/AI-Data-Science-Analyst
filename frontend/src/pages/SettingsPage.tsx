import React, { useState, useEffect } from 'react';
import { Settings, Database, Key, Server, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { connectionApi, mcpToolsApi } from '../services/api';
import toast from 'react-hot-toast';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<'general' | 'connections' | 'mcp' | 'about'>('general');
  const [connections, setConnections] = useState<any[]>([]);
  const [mcpTools, setMcpTools] = useState<any[]>([]);
  const [mcpAgents, setMcpAgents] = useState<any[]>([]);

  // New connection form
  const [connForm, setConnForm] = useState({
    name: '', engine: 'postgresql', host: '', port: 5432,
    database: '', username: '', password: '',
  });

  useEffect(() => {
    if (activeTab === 'connections') loadConnections();
    if (activeTab === 'mcp') loadMcpInfo();
  }, [activeTab]);

  async function loadConnections() {
    try {
      const res = await connectionApi.list();
      setConnections(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadMcpInfo() {
    try {
      const [toolsRes, agentsRes] = await Promise.allSettled([
        mcpToolsApi.listTools(),
        mcpToolsApi.listAgents(),
      ]);
      if (toolsRes.status === 'fulfilled') setMcpTools(toolsRes.value.data.tools || []);
      if (agentsRes.status === 'fulfilled') setMcpAgents(agentsRes.value.data.agents || []);
    } catch (e) { console.error(e); }
  }

  async function createConnection() {
    try {
      await connectionApi.create(connForm);
      toast.success('Connection created');
      loadConnections();
      setConnForm({ name: '', engine: 'postgresql', host: '', port: 5432, database: '', username: '', password: '' });
    } catch (e: any) { toast.error('Failed to create connection'); }
  }

  async function testConnection(id: string) {
    try {
      const res = await connectionApi.test(id);
      if (res.data.success) toast.success('Connection successful');
      else toast.error('Connection failed');
    } catch (e: any) { toast.error('Connection test failed'); }
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Settings</h1>

      <div className="flex space-x-4 mb-6">
        {(['general', 'connections', 'mcp', 'about'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize ${
              activeTab === tab ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600'
            }`}>
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'general' && (
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <h2 className="text-lg font-semibold mb-4">General Settings</h2>
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">API Configuration</h3>
              <p className="text-xs text-gray-500 mb-2">
                API keys are configured via environment variables in Docker.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-500">Backend URL</p>
                  <p className="text-sm font-mono">{process.env.REACT_APP_API_URL || 'http://172.168.1.95:8050'}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-500">MCP Server URL</p>
                  <p className="text-sm font-mono">{process.env.REACT_APP_MCP_URL || 'http://172.168.1.95:4050'}</p>
                </div>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Service Ports</h3>
              <div className="grid grid-cols-3 gap-3 text-sm">
                {[
                  { name: 'Frontend', port: '3050' },
                  { name: 'Backend API', port: '8050' },
                  { name: 'MCP Server', port: '4050' },
                  { name: 'PostgreSQL', port: '5450' },
                  { name: 'Redis', port: '6350' },
                  { name: 'ChromaDB', port: '6340' },
                  { name: 'Flower', port: '5550' },
                  { name: 'pgAdmin', port: '5460' },
                ].map(s => (
                  <div key={s.name} className="bg-gray-50 p-2 rounded text-center">
                    <p className="text-xs text-gray-500">{s.name}</p>
                    <p className="font-mono font-medium">{s.port}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'connections' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h2 className="text-lg font-semibold mb-4">Add Database Connection</h2>
            <div className="grid grid-cols-2 gap-4">
              <input value={connForm.name} onChange={(e) => setConnForm({...connForm, name: e.target.value})}
                placeholder="Connection name" className="px-3 py-2 border rounded-lg text-sm" />
              <select value={connForm.engine} onChange={(e) => setConnForm({...connForm, engine: e.target.value})}
                className="px-3 py-2 border rounded-lg text-sm">
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="sqlite">SQLite</option>
                <option value="mssql">MS SQL Server</option>
              </select>
              <input value={connForm.host} onChange={(e) => setConnForm({...connForm, host: e.target.value})}
                placeholder="Host" className="px-3 py-2 border rounded-lg text-sm" />
              <input type="number" value={connForm.port} onChange={(e) => setConnForm({...connForm, port: parseInt(e.target.value)})}
                placeholder="Port" className="px-3 py-2 border rounded-lg text-sm" />
              <input value={connForm.database} onChange={(e) => setConnForm({...connForm, database: e.target.value})}
                placeholder="Database name" className="px-3 py-2 border rounded-lg text-sm" />
              <input value={connForm.username} onChange={(e) => setConnForm({...connForm, username: e.target.value})}
                placeholder="Username" className="px-3 py-2 border rounded-lg text-sm" />
              <input type="password" value={connForm.password} onChange={(e) => setConnForm({...connForm, password: e.target.value})}
                placeholder="Password" className="px-3 py-2 border rounded-lg text-sm" />
              <button onClick={createConnection} className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm">
                Add Connection
              </button>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h2 className="text-lg font-semibold mb-4">Connections</h2>
            {connections.length > 0 ? connections.map((c: any) => (
              <div key={c.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg mb-2">
                <div>
                  <p className="font-medium text-sm">{c.name}</p>
                  <p className="text-xs text-gray-500">{c.engine}://{c.host}:{c.port}/{c.database}</p>
                </div>
                <button onClick={() => testConnection(c.id)}
                  className="px-3 py-1 text-xs bg-primary-50 text-primary-600 rounded-lg">
                  Test
                </button>
              </div>
            )) : <p className="text-sm text-gray-500">No connections configured</p>}
          </div>
        </div>
      )}

      {activeTab === 'mcp' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h2 className="text-lg font-semibold mb-4">MCP Tools</h2>
            {mcpTools.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {mcpTools.map((t: any) => (
                  <div key={t.name} className="p-3 bg-gray-50 rounded-lg">
                    <p className="font-medium text-sm font-mono">{t.name}</p>
                    <p className="text-xs text-gray-500 mt-1">{t.description}</p>
                  </div>
                ))}
              </div>
            ) : <p className="text-sm text-gray-500">Could not connect to MCP server</p>}
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h2 className="text-lg font-semibold mb-4">A2A Agents</h2>
            {mcpAgents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {mcpAgents.map((a: any) => (
                  <div key={a.id} className="p-3 bg-gray-50 rounded-lg">
                    <p className="font-medium text-sm">{a.name}</p>
                    <p className="text-xs text-gray-500 mt-1">{a.description}</p>
                    <div className="flex gap-1 mt-2 flex-wrap">
                      {a.capabilities?.map((c: string) => (
                        <span key={c} className="px-1.5 py-0.5 bg-blue-50 text-blue-600 text-xs rounded">{c}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : <p className="text-sm text-gray-500">Could not connect to A2A server</p>}
          </div>
        </div>
      )}

      {activeTab === 'about' && (
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <h2 className="text-lg font-semibold mb-4">About</h2>
          <p className="text-sm text-gray-600 mb-4">
            AI Data Science Analyst is a multi-agent AI system for data science and analytics,
            based on the LangChain Data Science Assistant project by Mohammed Z Hassan.
          </p>
          <div className="space-y-2 text-sm">
            <p><strong>Backend:</strong> Django 5.1 + PostgreSQL + Celery + Redis</p>
            <p><strong>Frontend:</strong> React 18 + TypeScript + Tailwind CSS</p>
            <p><strong>AI:</strong> LangChain + OpenAI GPT + ChromaDB (RAG)</p>
            <p><strong>Protocols:</strong> MCP (Model Context Protocol) + A2A (Agent-to-Agent)</p>
            <p><strong>Architecture:</strong> Multi-agent with specialized agents for DA/DS tasks</p>
          </div>
        </div>
      )}
    </div>
  );
}
