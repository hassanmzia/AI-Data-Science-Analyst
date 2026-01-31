import React, { useState, useEffect } from 'react';
import { FolderOpen, Plus, Archive, FileText, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { projectApi } from '../services/api';
import { Project } from '../types';
import toast from 'react-hot-toast';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newType, setNewType] = useState('general');
  const [loading, setLoading] = useState(true);
  const [generatingSummary, setGeneratingSummary] = useState(false);

  useEffect(() => { loadProjects(); }, []);

  async function loadProjects() {
    setLoading(true);
    try {
      const res = await projectApi.list();
      setProjects(res.data.results || res.data || []);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  }

  async function createProject() {
    if (!newName) return;
    try {
      const res = await projectApi.create({
        name: newName, description: newDesc, project_type: newType,
      });
      toast.success('Project created');
      loadProjects();
      setShowCreate(false);
      setNewName(''); setNewDesc('');
      setSelectedProject(res.data);
    } catch (e: any) {
      toast.error(`Failed: ${e.response?.data?.error || e.message}`);
    }
  }

  async function archiveProject(project: Project) {
    try {
      await projectApi.archive(project.id);
      toast.success('Project archived');
      loadProjects();
      setSelectedProject(null);
    } catch (e: any) { toast.error('Archive failed'); }
  }

  async function generateSummary(project: Project) {
    setGeneratingSummary(true);
    try {
      const res = await projectApi.generateSummary(project.id);
      toast.success('Summary generated');
      const updated = await projectApi.get(project.id);
      setSelectedProject(updated.data);
    } catch (e: any) { toast.error('Summary generation failed'); }
    finally { setGeneratingSummary(false); }
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Projects</h1>
          <p className="text-sm text-gray-500 mt-1">Organize your analyses into projects</p>
        </div>
        <button onClick={() => setShowCreate(!showCreate)}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 flex items-center">
          <Plus size={16} className="mr-2" /> New Project
        </button>
      </div>

      {showCreate && (
        <div className="bg-white rounded-xl p-6 border border-gray-200 mb-6">
          <h3 className="font-medium text-gray-900 mb-4">Create New Project</h3>
          <div className="space-y-3">
            <input value={newName} onChange={(e) => setNewName(e.target.value)}
              placeholder="Project name" className="w-full px-4 py-2 border rounded-lg text-sm" />
            <textarea value={newDesc} onChange={(e) => setNewDesc(e.target.value)}
              placeholder="Description" rows={2} className="w-full px-4 py-2 border rounded-lg text-sm resize-none" />
            <div className="flex space-x-3">
              <select value={newType} onChange={(e) => setNewType(e.target.value)}
                className="px-4 py-2 border rounded-lg text-sm">
                {['general', 'eda', 'ml', 'analytics', 'reporting', 'research'].map(t => (
                  <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                ))}
              </select>
              <button onClick={createProject} className="px-6 py-2 bg-primary-600 text-white rounded-lg text-sm">
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-3">
          {projects.map(p => (
            <div key={p.id} onClick={() => setSelectedProject(p)}
              className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                selectedProject?.id === p.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-gray-900 text-sm">{p.name}</h3>
                <span className={`px-2 py-0.5 text-xs rounded-full ${
                  p.status === 'active' ? 'bg-green-100 text-green-700' :
                  p.status === 'completed' ? 'bg-blue-100 text-blue-700' :
                  'bg-gray-100 text-gray-700'
                }`}>{p.status}</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">{p.project_type} | {p.analysis_count || 0} analyses</p>
            </div>
          ))}
          {projects.length === 0 && !loading && (
            <p className="text-center text-gray-500 py-6">No projects yet</p>
          )}
        </div>

        <div className="lg:col-span-2">
          {selectedProject ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">{selectedProject.name}</h3>
                <div className="flex space-x-2">
                  <button onClick={() => generateSummary(selectedProject)}
                    disabled={generatingSummary}
                    className="px-3 py-1.5 text-xs bg-primary-50 text-primary-600 rounded-lg hover:bg-primary-100">
                    {generatingSummary ? <Loader2 size={12} className="animate-spin" /> : <FileText size={12} className="mr-1 inline" />}
                    Generate Summary
                  </button>
                  <button onClick={() => archiveProject(selectedProject)}
                    className="px-3 py-1.5 text-xs bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-100">
                    <Archive size={12} className="mr-1 inline" /> Archive
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-600 mb-4">{selectedProject.description}</p>
              {selectedProject.summary && (
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">AI Summary</h4>
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown>{selectedProject.summary}</ReactMarkdown>
                  </div>
                </div>
              )}
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-2xl font-bold text-gray-900">{selectedProject.dataset_count || 0}</p>
                  <p className="text-xs text-gray-500">Datasets</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-2xl font-bold text-gray-900">{selectedProject.analysis_count || 0}</p>
                  <p className="text-xs text-gray-500">Analyses</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-2xl font-bold text-gray-900">{selectedProject.tags?.length || 0}</p>
                  <p className="text-xs text-gray-500">Tags</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <FolderOpen size={40} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Select a project to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
