import React, { useState, useEffect } from 'react';
import { FolderArchive, Search, Copy, BookTemplate, Loader2, Star } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { archiveApi } from '../services/api';
import { ArchivedProject, ProjectTemplate } from '../types';
import toast from 'react-hot-toast';

export default function ArchivePage() {
  const [archives, setArchives] = useState<ArchivedProject[]>([]);
  const [templates, setTemplates] = useState<ProjectTemplate[]>([]);
  const [selectedArchive, setSelectedArchive] = useState<ArchivedProject | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'archives' | 'templates'>('archives');
  const [loading, setLoading] = useState(true);
  const [newProjectName, setNewProjectName] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);

  useEffect(() => { loadArchives(); loadTemplates(); }, []);

  async function loadArchives() {
    setLoading(true);
    try {
      const res = await archiveApi.list();
      setArchives(res.data.results || res.data || []);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  }

  async function loadTemplates() {
    try {
      const res = await archiveApi.templates.list();
      setTemplates(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  async function searchArchives() {
    if (!searchQuery) { loadArchives(); return; }
    try {
      const res = await archiveApi.search({ query: searchQuery });
      setArchives(res.data);
    } catch (e) { console.error(e); }
  }

  async function createFromArchive() {
    if (!selectedArchive || !newProjectName) return;
    try {
      await archiveApi.createFromArchive(selectedArchive.id, { name: newProjectName });
      toast.success('New project created from archive');
      setShowCreateModal(false);
      setNewProjectName('');
    } catch (e: any) { toast.error('Failed to create project'); }
  }

  async function makeTemplate() {
    if (!selectedArchive) return;
    try {
      await archiveApi.makeTemplate(selectedArchive.id);
      toast.success('Template created');
      loadTemplates();
    } catch (e: any) { toast.error('Failed to create template'); }
  }

  async function useTemplate(templateId: string) {
    const name = prompt('Project name:');
    if (!name) return;
    try {
      await archiveApi.templates.useTemplate(templateId, { name });
      toast.success('Project created from template');
    } catch (e: any) { toast.error('Failed'); }
  }

  const filtered = archives.filter(a =>
    a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    a.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Archive</h1>
      <p className="text-sm text-gray-500 mb-6">Browse archived projects and templates for reuse</p>

      {/* Tabs */}
      <div className="flex space-x-4 mb-6">
        <button onClick={() => setActiveTab('archives')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            activeTab === 'archives' ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600'
          }`}>
          Archived Projects ({archives.length})
        </button>
        <button onClick={() => setActiveTab('templates')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            activeTab === 'templates' ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600'
          }`}>
          Templates ({templates.length})
        </button>
      </div>

      {activeTab === 'archives' ? (
        <>
          {/* Search */}
          <div className="flex space-x-3 mb-6">
            <div className="flex-1 relative">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && searchArchives()}
                placeholder="Search archives..." className="w-full pl-10 pr-4 py-2 border rounded-lg text-sm" />
            </div>
            <button onClick={searchArchives} className="px-4 py-2 bg-gray-100 rounded-lg text-sm">Search</button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="space-y-3">
              {filtered.map(a => (
                <div key={a.id} onClick={() => setSelectedArchive(a)}
                  className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                    selectedArchive?.id === a.id ? 'border-primary-500' : 'border-gray-200'
                  }`}>
                  <h3 className="font-medium text-gray-900 text-sm">{a.name}</h3>
                  <p className="text-xs text-gray-500 mt-1">
                    {a.category} | {a.analysis_count || 0} analyses | Used {a.use_count}x
                  </p>
                  {a.tags?.length > 0 && (
                    <div className="flex gap-1 mt-2 flex-wrap">
                      {a.tags.slice(0, 3).map(tag => (
                        <span key={tag} className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {filtered.length === 0 && <p className="text-center text-gray-500 py-6">No archived projects</p>}
            </div>

            <div className="lg:col-span-2">
              {selectedArchive ? (
                <div className="bg-white rounded-xl border border-gray-200 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">{selectedArchive.name}</h3>
                    <div className="flex space-x-2">
                      <button onClick={() => setShowCreateModal(true)}
                        className="px-3 py-1.5 text-xs bg-primary-50 text-primary-600 rounded-lg hover:bg-primary-100 flex items-center">
                        <Copy size={12} className="mr-1" /> Reuse
                      </button>
                      <button onClick={makeTemplate}
                        className="px-3 py-1.5 text-xs bg-green-50 text-green-600 rounded-lg hover:bg-green-100 flex items-center">
                        <BookTemplate size={12} className="mr-1" /> Make Template
                      </button>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{selectedArchive.description}</p>
                  {selectedArchive.summary && (
                    <div className="bg-gray-50 rounded-lg p-4 mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Summary</h4>
                      <div className="prose prose-sm max-w-none">
                        <ReactMarkdown>{selectedArchive.summary}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                  {selectedArchive.methodology && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-1">Methodology</h4>
                      <p className="text-sm text-gray-600">{selectedArchive.methodology}</p>
                    </div>
                  )}
                  {selectedArchive.tools_used?.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-1">Tools Used</h4>
                      <div className="flex gap-1 flex-wrap">
                        {selectedArchive.tools_used.map(t => (
                          <span key={t} className="px-2 py-0.5 bg-blue-50 text-blue-600 text-xs rounded-full">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Create from archive modal */}
                  {showCreateModal && (
                    <div className="mt-4 border-t pt-4">
                      <h4 className="text-sm font-medium mb-2">Create new project from this archive:</h4>
                      <div className="flex space-x-3">
                        <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)}
                          placeholder="New project name" className="flex-1 px-3 py-2 border rounded-lg text-sm" />
                        <button onClick={createFromArchive}
                          className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm">Create</button>
                        <button onClick={() => setShowCreateModal(false)}
                          className="px-4 py-2 bg-gray-100 rounded-lg text-sm">Cancel</button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
                  <FolderArchive size={40} className="mx-auto text-gray-300 mb-3" />
                  <p className="text-gray-500">Select an archived project to view details and reuse</p>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        /* Templates tab */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map(t => (
            <div key={t.id} className="bg-white rounded-xl p-5 border border-gray-200">
              <h3 className="font-medium text-gray-900">{t.name}</h3>
              <p className="text-sm text-gray-500 mt-1">{t.description}</p>
              <p className="text-xs text-gray-400 mt-2">
                {t.steps?.length || 0} steps | Used {t.use_count}x
              </p>
              {t.sample_queries?.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-gray-500 font-medium">Sample queries:</p>
                  {t.sample_queries.slice(0, 2).map((q, i) => (
                    <p key={i} className="text-xs text-gray-400 mt-1 truncate">{q}</p>
                  ))}
                </div>
              )}
              <button onClick={() => useTemplate(t.id)}
                className="mt-4 w-full px-3 py-2 bg-primary-50 text-primary-600 rounded-lg text-sm hover:bg-primary-100">
                Use Template
              </button>
            </div>
          ))}
          {templates.length === 0 && (
            <p className="col-span-3 text-center text-gray-500 py-10">
              No templates yet. Archive a project and convert it to a template.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
