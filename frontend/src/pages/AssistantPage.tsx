import React, { useState, useEffect, useRef } from 'react';
import {
  Send, Plus, Trash2, Bot, User, Loader2,
  Database, FileText, Settings2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import Plot from 'react-plotly.js';
import { agentApi, datasetApi, documentApi } from '../services/api';
import { Conversation, Message, Dataset, Document } from '../types';
import toast from 'react-hot-toast';

const ASSISTANT_TYPES = [
  { value: 'general', label: 'General Assistant', desc: 'All-purpose data science help' },
  { value: 'data_analyst', label: 'Data Analyst', desc: 'EDA, stats, data insights' },
  { value: 'data_scientist', label: 'Data Scientist', desc: 'ML, hypothesis testing' },
  { value: 'sql_expert', label: 'SQL Expert', desc: 'Natural language SQL' },
  { value: 'ml_engineer', label: 'ML Engineer', desc: 'Model building & training' },
  { value: 'rag_assistant', label: 'RAG Assistant', desc: 'Document Q&A' },
];

export default function AssistantPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedDocument, setSelectedDocument] = useState<string>('');
  const [assistantType, setAssistantType] = useState('general');
  const [showConfig, setShowConfig] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConversations();
    loadDatasets();
    loadDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function loadConversations() {
    try {
      const res = await agentApi.conversations.list();
      setConversations(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadDatasets() {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadDocuments() {
    try {
      const res = await documentApi.list();
      setDocuments(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function loadMessages(convId: string) {
    try {
      const res = await agentApi.conversations.messages(convId);
      setMessages(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function createConversation() {
    try {
      const res = await agentApi.conversations.create({
        title: 'New Chat',
        assistant_type: assistantType,
        dataset: selectedDataset || undefined,
      });
      const conv = res.data;
      setConversations(prev => [conv, ...prev]);
      setActiveConvId(conv.id);
      setMessages([]);
    } catch (e: any) {
      toast.error('Failed to create conversation');
    }
  }

  async function selectConversation(conv: Conversation) {
    setActiveConvId(conv.id);
    setAssistantType(conv.assistant_type);
    await loadMessages(conv.id);
  }

  async function sendMessage() {
    if (!input.trim() || !activeConvId || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Optimistic update
    const tempUserMsg: Message = {
      id: 'temp-user-' + Date.now(),
      conversation: activeConvId,
      role: 'user',
      content: userMessage,
      agent_name: '',
      tool_calls: [],
      code_blocks: [],
      visualizations: [],
      metadata: {},
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const res = await agentApi.conversations.chat(activeConvId, {
        message: userMessage,
        dataset_id: selectedDataset || undefined,
        document_id: selectedDocument || undefined,
      });

      const { user_message, assistant_message } = res.data;
      setMessages(prev => {
        const filtered = prev.filter(m => m.id !== tempUserMsg.id);
        return [...filtered, user_message, assistant_message];
      });

      // Update conversation title if first message
      if (messages.length === 0) {
        loadConversations();
      }
    } catch (e: any) {
      toast.error('Failed to send message');
      setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id));
    } finally {
      setLoading(false);
    }
  }

  async function deleteConversation(convId: string) {
    try {
      await agentApi.conversations.delete(convId);
      setConversations(prev => prev.filter(c => c.id !== convId));
      if (activeConvId === convId) {
        setActiveConvId(null);
        setMessages([]);
      }
      toast.success('Conversation deleted');
    } catch (e) { toast.error('Failed to delete'); }
  }

  return (
    <div className="flex h-full">
      {/* Conversation Sidebar */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={createConversation}
            className="w-full flex items-center justify-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
          >
            <Plus size={16} className="mr-2" /> New Chat
          </button>
        </div>

        {/* Config panel */}
        <div className="p-3 border-b border-gray-200">
          <button onClick={() => setShowConfig(!showConfig)} className="flex items-center text-sm text-gray-600">
            <Settings2 size={14} className="mr-1" /> Configuration
          </button>
          {showConfig && (
            <div className="mt-2 space-y-2">
              <select
                value={assistantType}
                onChange={(e) => setAssistantType(e.target.value)}
                className="w-full text-xs p-2 border rounded-lg"
              >
                {ASSISTANT_TYPES.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full text-xs p-2 border rounded-lg"
              >
                <option value="">No dataset</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
              <select
                value={selectedDocument}
                onChange={(e) => setSelectedDocument(e.target.value)}
                className="w-full text-xs p-2 border rounded-lg"
              >
                <option value="">No document</option>
                {documents.map(d => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto">
          {conversations.map(conv => (
            <div
              key={conv.id}
              onClick={() => selectConversation(conv)}
              className={`flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-50 border-b border-gray-100 ${
                activeConvId === conv.id ? 'bg-primary-50' : ''
              }`}
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{conv.title}</p>
                <p className="text-xs text-gray-500">{conv.assistant_type}</p>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                className="p-1 hover:bg-red-50 rounded"
              >
                <Trash2 size={14} className="text-gray-400 hover:text-red-500" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {activeConvId ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 && (
                <div className="text-center py-20">
                  <Bot size={48} className="mx-auto text-gray-300 mb-4" />
                  <h3 className="text-lg font-medium text-gray-600">Start a conversation</h3>
                  <p className="text-sm text-gray-400 mt-2">
                    Ask me anything about data analysis, ML, statistics, or SQL
                  </p>
                  <div className="mt-6 grid grid-cols-2 gap-3 max-w-lg mx-auto">
                    {[
                      'How many rows are in my dataset?',
                      'Create a correlation heatmap',
                      'Train a logistic regression model',
                      'Perform a t-test on age vs outcome',
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => { setInput(q); }}
                        className="p-3 text-left text-xs bg-gray-50 rounded-lg hover:bg-gray-100 text-gray-600"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-3xl rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white border border-gray-200 text-gray-800'
                  }`}>
                    {msg.role === 'assistant' && msg.agent_name && (
                      <p className="text-xs text-primary-500 font-medium mb-1">
                        {msg.agent_name}
                      </p>
                    )}
                    <div className="text-sm prose prose-sm max-w-none">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.role === 'assistant' && msg.visualizations && msg.visualizations.length > 0 && (
                      <div className="mt-3 space-y-3">
                        {msg.visualizations.map((viz: any, idx: number) => (
                          viz?.plotly_json?.data && (
                            <Plot
                              key={idx}
                              data={viz.plotly_json.data}
                              layout={{
                                ...(viz.plotly_json.layout || {}),
                                autosize: true,
                                margin: { t: 50, b: 50, l: 50, r: 30 },
                              }}
                              config={{ responsive: true, displayModeBar: true }}
                              style={{ width: '100%', height: '380px' }}
                              useResizeHandler
                            />
                          )
                        ))}
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.metadata?.plotly_json?.data && (
                      <div className="mt-3">
                        <Plot
                          data={msg.metadata.plotly_json.data}
                          layout={{
                            ...(msg.metadata.plotly_json.layout || {}),
                            autosize: true,
                            margin: { t: 50, b: 50, l: 50, r: 30 },
                          }}
                          config={{ responsive: true, displayModeBar: true }}
                          style={{ width: '100%', height: '380px' }}
                          useResizeHandler
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start animate-fade-in">
                  <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-200 p-4 bg-white">
              <div className="flex items-center space-x-3 max-w-4xl mx-auto">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                    placeholder="Ask about your data..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                    disabled={loading}
                  />
                </div>
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || loading}
                  className="p-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                </button>
              </div>
              <div className="flex items-center justify-center mt-2 space-x-4 text-xs text-gray-400">
                {selectedDataset && (
                  <span className="flex items-center">
                    <Database size={12} className="mr-1" /> Dataset active
                  </span>
                )}
                {selectedDocument && (
                  <span className="flex items-center">
                    <FileText size={12} className="mr-1" /> Document active
                  </span>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Bot size={64} className="mx-auto text-gray-300 mb-4" />
              <h2 className="text-xl font-semibold text-gray-600">AI Data Science Assistant</h2>
              <p className="text-gray-400 mt-2">Select a conversation or start a new one</p>
              <button
                onClick={createConversation}
                className="mt-4 px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm"
              >
                New Conversation
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
