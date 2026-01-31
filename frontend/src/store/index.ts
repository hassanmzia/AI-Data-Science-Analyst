import { create } from 'zustand';
import { Dataset, Conversation, Project, AnalysisSession, ArchivedProject } from '../types';

interface AppState {
  // Active dataset
  activeDataset: Dataset | null;
  setActiveDataset: (dataset: Dataset | null) => void;

  // Active conversation
  activeConversation: Conversation | null;
  setActiveConversation: (conv: Conversation | null) => void;

  // Active project
  activeProject: Project | null;
  setActiveProject: (project: Project | null) => void;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Theme
  darkMode: boolean;
  toggleDarkMode: () => void;

  // Notifications
  notifications: Array<{ id: string; type: string; message: string }>;
  addNotification: (notification: { type: string; message: string }) => void;
  removeNotification: (id: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeDataset: null,
  setActiveDataset: (dataset) => set({ activeDataset: dataset }),

  activeConversation: null,
  setActiveConversation: (conv) => set({ activeConversation: conv }),

  activeProject: null,
  setActiveProject: (project) => set({ activeProject: project }),

  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  darkMode: false,
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),

  notifications: [],
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: Date.now().toString() },
      ],
    })),
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}));
