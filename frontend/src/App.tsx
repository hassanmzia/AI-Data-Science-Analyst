import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useAppStore } from './store';
import {
  LayoutDashboard, Database, MessageSquare, FlaskConical,
  BarChart3, BrainCircuit, FileSearch, FolderArchive, Settings,
  Menu, X, Bot
} from 'lucide-react';

// Pages
import Dashboard from './pages/Dashboard';
import DatasetsPage from './pages/DatasetsPage';
import AssistantPage from './pages/AssistantPage';
import AnalysisPage from './pages/AnalysisPage';
import VisualizationPage from './pages/VisualizationPage';
import MLModelsPage from './pages/MLModelsPage';
import DocumentsPage from './pages/DocumentsPage';
import ProjectsPage from './pages/ProjectsPage';
import ArchivePage from './pages/ArchivePage';
import SettingsPage from './pages/SettingsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30000, retry: 1 },
  },
});

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/assistant', label: 'AI Assistant', icon: Bot },
  { path: '/datasets', label: 'Datasets', icon: Database },
  { path: '/analysis', label: 'Analysis', icon: FlaskConical },
  { path: '/visualization', label: 'Visualization', icon: BarChart3 },
  { path: '/ml-models', label: 'ML Models', icon: BrainCircuit },
  { path: '/documents', label: 'Documents (RAG)', icon: FileSearch },
  { path: '/projects', label: 'Projects', icon: FolderArchive },
  { path: '/archive', label: 'Archive', icon: FolderArchive },
  { path: '/settings', label: 'Settings', icon: Settings },
];

function AppLayout() {
  const { sidebarOpen, toggleSidebar } = useAppStore();

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-white border-r border-gray-200 transition-all duration-300 flex flex-col`}>
        {/* Logo */}
        <div className="h-16 flex items-center px-4 border-b border-gray-200">
          <button onClick={toggleSidebar} className="p-1 rounded-lg hover:bg-gray-100">
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          {sidebarOpen && (
            <span className="ml-3 font-bold text-lg text-primary-700">DS Analyst</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center px-4 py-3 mx-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <item.icon size={20} className="flex-shrink-0" />
              {sidebarOpen && <span className="ml-3 text-sm">{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Version */}
        {sidebarOpen && (
          <div className="p-4 border-t border-gray-200">
            <p className="text-xs text-gray-400">AI Data Science Analyst v1.0</p>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/assistant" element={<AssistantPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/visualization" element={<VisualizationPage />} />
          <Route path="/ml-models" element={<MLModelsPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/archive" element={<ArchivePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppLayout />
        <Toaster position="top-right" />
      </Router>
    </QueryClientProvider>
  );
}

export default App;
