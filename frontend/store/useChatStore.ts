import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { memoryService } from '../services/memoryService';
import { apiClient } from '../services/apiClient';
import { Message, Citation } from '../types';

// Re-export types so store/index.ts can re-export them
export type { Message, Citation };

// ============================================================================
// Types
// ============================================================================

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  lastMessage?: string;
  model?: ModelType;
  isPinned?: boolean;
  isArchived?: boolean;
  messages: Message[];
}

export type ModelType = 'medgemma' | 'ollama' | 'gemini';

export interface ChatSettings {
  temperature: number;
  systemPrompt: string;
  maxTokens: number;
  streamResponses: boolean;
  autoGenerateTitle: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
const generateSessionId = () => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

/** Generate a short title from the first user message */
function generateTitleFromMessage(content: string): string {
  let text = content.replace(/\n/g, ' ').trim();
  text = text.replace(/^(hi|hello|hey|good morning|good afternoon|good evening|yo|sup)[,!.\s]*/i, '').trim();
  if (!text) return content.slice(0, 40);
  text = text.charAt(0).toUpperCase() + text.slice(1);
  if (text.length > 40) {
    text = text.slice(0, 40).replace(/\s+\S*$/, '') + '…';
  }
  return text;
}

/** Group sessions by date category */
export function groupSessionsByDate(sessions: ChatSession[]): Record<string, ChatSession[]> {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const prev7 = new Date(today.getTime() - 7 * 86400000);
  const prev30 = new Date(today.getTime() - 30 * 86400000);

  const groups: Record<string, ChatSession[]> = {};

  for (const s of sessions) {
    const d = new Date(s.updatedAt);
    let key: string;
    if (d >= today) key = 'Today';
    else if (d >= yesterday) key = 'Yesterday';
    else if (d >= prev7) key = 'Previous 7 Days';
    else if (d >= prev30) key = 'Previous 30 Days';
    else key = 'Older';

    if (!groups[key]) groups[key] = [];
    groups[key].push(s);
  }

  return groups;
}

const defaultMessage: Message = {
  id: 'welcome',
  role: 'assistant',
  content: "Hello! I'm your AI health assistant. How can I help you today with your cardiovascular health?",
  timestamp: new Date().toISOString(),
};

const defaultSettings: ChatSettings = {
  temperature: 0.7,
  systemPrompt: 'You are Cardio AI, a helpful cardiovascular health assistant. Be concise, caring, and medically informed. Always recommend consulting a doctor for serious concerns.',
  maxTokens: 2048,
  streamResponses: true,
  autoGenerateTitle: true,
};

// ============================================================================
// State Interface
// ============================================================================

export interface ChatState {
  sessions: ChatSession[];
  currentSessionId: string | null;
  messages: Message[];

  isLoading: boolean;
  isStreaming: boolean;
  isSearchingMemories: boolean;
  error: string | null;

  selectedModel: ModelType;
  isThinkingEnabled: boolean;
  autoSaveEnabled: boolean;
  settings: ChatSettings;

  isRecording: boolean;
  isPlayingId: string | null;

  // Message Actions
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  removeMessage: (id: string) => void;
  clearMessages: () => void;

  // Session Actions
  createSession: (title?: string) => string;
  loadSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  updateSessionTitle: (sessionId: string, title: string) => Promise<void>;
  loadSessions: (userId: string) => Promise<void>;
  pinSession: (sessionId: string) => void;
  archiveSession: (sessionId: string) => void;
  duplicateSession: (sessionId: string) => string;
  deleteAllSessions: () => void;

  // Settings
  updateSettings: (updates: Partial<ChatSettings>) => void;

  // State Setters
  setLoading: (loading: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  setSearchingMemories: (searching: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedModel: (model: ModelType) => void;
  setThinkingEnabled: (enabled: boolean) => void;
  setRecording: (recording: boolean) => void;
  setPlayingId: (id: string | null) => void;

  // Computed
  getCurrentSession: () => ChatSession | null;
  getSessionMessages: (sessionId: string) => Message[];
  searchSessions: (query: string) => ChatSession[];
}

// ============================================================================
// Store
// ============================================================================

export const useChatStore = create<ChatState>()(
  devtools(
    persist(
      immer((set, get) => ({
        sessions: [] as ChatSession[],
        currentSessionId: null as string | null,
        messages: [defaultMessage] as Message[],

        isLoading: false,
        isStreaming: false,
        isSearchingMemories: false,
        error: null as string | null,

        selectedModel: 'medgemma' as ModelType,
        isThinkingEnabled: false,
        autoSaveEnabled: true,
        settings: { ...defaultSettings },

        isRecording: false,
        isPlayingId: null as string | null,

        // ====================================================================
        // Message Actions
        // ====================================================================

        setMessages: (messages: Message[]) =>
          set((state) => {
            state.messages = messages;
            const session = state.sessions.find((s) => s.id === state.currentSessionId);
            if (session) {
              session.messages = messages;
              session.messageCount = messages.length;
            }
          }),

        addMessage: (message: Message) =>
          set((state) => {
            const msg = {
              ...message,
              id: message.id || generateId(),
              timestamp: message.timestamp || new Date().toISOString(),
            };
            state.messages.push(msg);

            const session = state.sessions.find((s) => s.id === state.currentSessionId);
            if (session) {
              session.messages = [...state.messages];
              session.updatedAt = new Date().toISOString();
              session.messageCount = state.messages.length;
              session.lastMessage = msg.content.slice(0, 100);

              // Auto-generate title from first user message
              if (
                state.settings.autoGenerateTitle &&
                msg.role === 'user' &&
                (session.title.startsWith('New Chat') || session.title.startsWith('Chat '))
              ) {
                session.title = generateTitleFromMessage(msg.content);
              }
            }
          }),

        updateMessage: (id: string, updates: Partial<Message>) =>
          set((state) => {
            const idx = state.messages.findIndex((m) => m.id === id);
            if (idx !== -1) {
              state.messages[idx] = { ...state.messages[idx], ...updates };
            }
            const session = state.sessions.find((s) => s.id === state.currentSessionId);
            if (session) {
              session.messages = [...state.messages];
              if (updates.content) {
                session.lastMessage = updates.content.slice(0, 100);
              }
            }
          }),

        removeMessage: (id: string) =>
          set((state) => {
            state.messages = state.messages.filter((m) => m.id !== id);
            const session = state.sessions.find((s) => s.id === state.currentSessionId);
            if (session) {
              session.messages = [...state.messages];
              session.messageCount = state.messages.length;
            }
          }),

        clearMessages: () =>
          set((state) => {
            state.messages = [defaultMessage];
            const session = state.sessions.find((s) => s.id === state.currentSessionId);
            if (session) {
              session.messages = [defaultMessage];
              session.messageCount = 1;
            }
          }),

        // ====================================================================
        // Session Actions
        // ====================================================================

        createSession: (title?: string) => {
          const sessionId = generateSessionId();
          const now = new Date().toISOString();

          set((state) => {
            // Save current session before switching
            const currentSession = state.sessions.find((s) => s.id === state.currentSessionId);
            if (currentSession) {
              currentSession.messages = [...state.messages];
            }

            const newSession: ChatSession = {
              id: sessionId,
              title: title || 'New Chat',
              createdAt: now,
              updatedAt: now,
              messageCount: 1,
              model: state.selectedModel,
              isPinned: false,
              isArchived: false,
              messages: [defaultMessage],
            };

            state.sessions.unshift(newSession);
            state.currentSessionId = sessionId;
            state.messages = [defaultMessage];
          });

          return sessionId;
        },

        loadSession: (sessionId: string) => {
          set((state) => {
            // Save current session before switching
            const currentSession = state.sessions.find((s) => s.id === state.currentSessionId);
            if (currentSession) {
              currentSession.messages = [...state.messages];
            }

            const target = state.sessions.find((s) => s.id === sessionId);
            if (target) {
              state.currentSessionId = sessionId;
              state.messages =
                target.messages && target.messages.length > 0
                  ? [...target.messages]
                  : [defaultMessage];
            }
          });
        },

        deleteSession: (sessionId: string) => {
          set((state) => {
            state.sessions = state.sessions.filter((s) => s.id !== sessionId);
            if (state.currentSessionId === sessionId) {
              if (state.sessions.length > 0) {
                const next = state.sessions[0];
                state.currentSessionId = next.id;
                state.messages =
                  next.messages && next.messages.length > 0 ? [...next.messages] : [defaultMessage];
              } else {
                state.currentSessionId = null;
                state.messages = [defaultMessage];
              }
            }
          });
          memoryService.deleteSession(sessionId).catch(() => {});
        },

        updateSessionTitle: async (sessionId: string, title: string) => {
          set((state) => {
            const session = state.sessions.find((s) => s.id === sessionId);
            if (session) session.title = title;
          });
          memoryService.updateSession(sessionId, { title }).catch(() => {});
        },

        loadSessions: async (userId: string) => {
          try {
            const sessions = await memoryService.getSessions(userId);
            set((state) => {
              const localIds = new Set(state.sessions.map((s) => s.id));
              const backendSessions: ChatSession[] = sessions
                .filter((s) => !localIds.has(s.sessionId))
                .map((s) => ({
                  id: s.sessionId,
                  title: `Chat ${new Date(s.createdAt || Date.now()).toLocaleDateString()}`,
                  createdAt: s.createdAt || new Date().toISOString(),
                  updatedAt: s.lastActivity || new Date().toISOString(),
                  messageCount: s.messageCount,
                  model: 'medgemma' as ModelType,
                  isPinned: false,
                  isArchived: false,
                  messages: [],
                }));
              state.sessions = [...state.sessions, ...backendSessions];
            });
          } catch (error) {
            console.error('Failed to load sessions from backend:', error);
          }
        },

        pinSession: (sessionId: string) =>
          set((state) => {
            const session = state.sessions.find((s) => s.id === sessionId);
            if (session) session.isPinned = !session.isPinned;
          }),

        archiveSession: (sessionId: string) =>
          set((state) => {
            const session = state.sessions.find((s) => s.id === sessionId);
            if (session) session.isArchived = !session.isArchived;
          }),

        duplicateSession: (sessionId: string) => {
          const newId = generateSessionId();
          set((state) => {
            const source = state.sessions.find((s) => s.id === sessionId);
            if (source) {
              state.sessions.unshift({
                ...source,
                id: newId,
                title: `${source.title} (Copy)`,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                isPinned: false,
                messages: [...source.messages],
              });
            }
          });
          return newId;
        },

        deleteAllSessions: () =>
          set((state) => {
            state.sessions = [];
            state.currentSessionId = null;
            state.messages = [defaultMessage];
          }),

        // ====================================================================
        // Settings
        // ====================================================================

        updateSettings: (updates: Partial<ChatSettings>) =>
          set((state) => {
            state.settings = { ...state.settings, ...updates };
          }),

        // ====================================================================
        // State Setters
        // ====================================================================

        setLoading: (loading: boolean) => set({ isLoading: loading }),
        setStreaming: (streaming: boolean) => set({ isStreaming: streaming }),
        setSearchingMemories: (searching: boolean) => set({ isSearchingMemories: searching }),
        setError: (error: string | null) => set({ error }),
        setSelectedModel: (model: ModelType) => set({ selectedModel: model }),
        setThinkingEnabled: (enabled: boolean) => set({ isThinkingEnabled: enabled }),
        setRecording: (recording: boolean) => set({ isRecording: recording }),
        setPlayingId: (id: string | null) => set({ isPlayingId: id }),

        // ====================================================================
        // Computed
        // ====================================================================

        getCurrentSession: () => {
          const state = get();
          return state.sessions.find((s) => s.id === state.currentSessionId) || null;
        },

        getSessionMessages: (sessionId: string) => {
          const state = get();
          const session = state.sessions.find((s) => s.id === sessionId);
          return session?.messages || [];
        },

        searchSessions: (query: string) => {
          const state = get();
          const q = query.toLowerCase();
          return state.sessions.filter(
            (s) =>
              !s.isArchived &&
              (s.title.toLowerCase().includes(q) ||
                s.lastMessage?.toLowerCase().includes(q) ||
                s.messages.some((m) => m.content.toLowerCase().includes(q)))
          );
        },
      })),
      {
        name: 'chat-store',
        version: 2,
        partialize: (state) => ({
          sessions: state.sessions,
          currentSessionId: state.currentSessionId,
          selectedModel: state.selectedModel,
          isThinkingEnabled: state.isThinkingEnabled,
          autoSaveEnabled: state.autoSaveEnabled,
          settings: state.settings,
        }),
        migrate: (persistedState: any, version: number) => {
          if (version < 2) {
            const sessions = (persistedState as any).sessions || [];
            return {
              ...persistedState,
              sessions: sessions.map((s: any) => ({
                ...s,
                messages: s.messages || [],
                isPinned: s.isPinned || false,
                isArchived: s.isArchived || false,
              })),
              settings: defaultSettings,
            };
          }
          return persistedState;
        },
      }
    ),
    { name: 'ChatStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectMessages = (state: ChatState) => state.messages;
export const selectIsLoading = (state: ChatState) => state.isLoading;
export const selectIsStreaming = (state: ChatState) => state.isStreaming;
export const selectSelectedModel = (state: ChatState) => state.selectedModel;
export const selectSessions = (state: ChatState) => state.sessions;
export const selectCurrentSession = (state: ChatState) =>
  state.sessions.find((s) => s.id === state.currentSessionId) || null;

// ============================================================================
// Actions (callable outside React)
// ============================================================================

export type SearchMode = 'default' | 'web_search' | 'deep_search' | 'memory';

// Module-level AbortController for cancelling in-flight AI requests
let currentAbortController: AbortController | null = null;

export const chatActions = {
  /** Abort the current AI request if one is in progress */
  stopGeneration: () => {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
    const store = useChatStore.getState();
    // Mark any streaming message as finished
    const streamingMsg = store.messages.find(m => m.isStreaming);
    if (streamingMsg) {
      store.updateMessage(streamingMsg.id, {
        isStreaming: false,
        content: streamingMsg.content || '*(Generation stopped)*',
      });
    }
    store.setLoading(false);
    store.setStreaming(false);
  },

  sendMessage: async (content: string, model?: ModelType, searchMode: SearchMode = 'default') => {
    const store = useChatStore.getState();
    const userId = localStorage.getItem('user_id');
    if (!userId) {
      console.error('User ID not found. Please log in.');
      return;
    }

    const sessionId = store.currentSessionId || store.createSession();

    const userMessage: Message & { searchMode?: SearchMode } = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
      ...(searchMode !== 'default' && { searchMode }),
    };
    store.addMessage(userMessage);
    store.setLoading(true);
    store.setError(null);

    // Create a new AbortController for this request
    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;

    try {
      const selectedModel = model || store.selectedModel;
      // Frontend guard: route all chat requests through the MedGemma backend path.
      const effectiveModel: ModelType = selectedModel === 'gemini' ? 'medgemma' : selectedModel;
      const { settings } = store;

      const assistantId = generateId();
      store.addMessage({
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
      });

      store.setStreaming(true);

      if (effectiveModel === 'ollama' || effectiveModel === 'medgemma') {
        const conversationHistory = store.messages
          .filter((m) => m.id !== 'welcome' && m.id !== assistantId)
          .map((m) => ({ role: m.role, content: m.content }));

        const searchModePrompt = searchMode === 'web_search'
          ? '\n[INSTRUCTION: Include relevant web search results and cite sources in your response.]'
          : searchMode === 'deep_search'
          ? '\n[INSTRUCTION: Provide a thorough, detailed, in-depth analysis with comprehensive explanations.]'
          : searchMode === 'memory'
          ? '\n[INSTRUCTION: Focus on referencing the patient\'s health history and previous conversations for context.]'
          : '';

        const fullHistory = [
          { role: 'system' as const, content: settings.systemPrompt + searchModePrompt },
          ...conversationHistory,
          { role: 'user' as const, content },
        ];

        const generator = apiClient.streamOllamaResponse({
          message: content,
          conversation_history: fullHistory,
          temperature: settings.temperature,
          signal,
          web_search: searchMode === 'web_search',
          deep_search: searchMode === 'deep_search',
          thinking: searchMode === 'deep_search', // Enable thinking for deep search
        });

        // Show a progress indicator while waiting for long operations
        const isLongOperation = searchMode === 'web_search' || searchMode === 'deep_search';
        let thinkingInterval: ReturnType<typeof setInterval> | null = null;
        if (isLongOperation) {
          const thinkingSteps = [
            searchMode === 'web_search' ? '🔍 Searching the web...' : '🧠 Performing deep analysis...',
            searchMode === 'web_search' ? '🌐 Gathering search results...' : '📊 Analyzing information...',
            searchMode === 'web_search' ? '📝 Compiling findings...' : '🔬 Deep research in progress...',
            '⏳ Still working — this may take a moment...',
          ];
          let stepIndex = 0;
          store.updateMessage(assistantId, { content: thinkingSteps[0] });
          thinkingInterval = setInterval(() => {
            if (signal.aborted) {
              if (thinkingInterval) clearInterval(thinkingInterval);
              return;
            }
            stepIndex = Math.min(stepIndex + 1, thinkingSteps.length - 1);
            // Only update if we haven't started receiving real content yet
            if (!fullContent) {
              store.updateMessage(assistantId, { content: thinkingSteps[stepIndex] });
            }
          }, 8000); // Update every 8 seconds
        }

        let fullContent = '';
        for await (const chunk of generator) {
          // Check if generation was stopped
          if (signal.aborted) break;
          if (chunk.type === 'token') {
            // Clear the thinking indicator once real content arrives
            if (thinkingInterval && !fullContent) {
              clearInterval(thinkingInterval);
              thinkingInterval = null;
            }
            fullContent += chunk.data;
            store.updateMessage(assistantId, { content: fullContent });
          } else if (chunk.type === 'error') {
            if (thinkingInterval) clearInterval(thinkingInterval);
            throw new Error(chunk.data.error);
          }
        }

        if (thinkingInterval) clearInterval(thinkingInterval);

        if (!signal.aborted) {
          store.updateMessage(assistantId, { isStreaming: false });
        }
      } else {
        const response = await memoryService.aiQuery(userId, sessionId, content, {
          aiProvider: 'gemini',
          patientName: 'User',
          searchMode: searchMode !== 'default' ? searchMode : undefined,
          signal,
        });

        if (response.success) {
          store.updateMessage(assistantId, {
            content: response.response,
            isStreaming: false,
            metadata: {
              model: 'gemini',
              processingTime: response.metadata.processingTimeMs,
              tokens: response.metadata.tokensEstimated,
              memoryContext: response.contextUsed.map((c) => c.source),
            },
          });
        } else {
          throw new Error(response.error || 'AI query failed');
        }
      }
    } catch (error) {
      // If aborted (user hit stop), don't show as error
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      const errorMsg = error instanceof Error ? error.message : 'Failed to send message';
      store.setError(errorMsg);

      // Determine user-friendly error message
      let displayMessage = 'Sorry, I encountered an error processing your request.';
      if (errorMsg.includes('401') || errorMsg.toLowerCase().includes('session expired') || errorMsg.toLowerCase().includes('unauthorized')) {
        displayMessage = '🔒 Your session has expired. Please log in again to continue chatting.';
      } else if (errorMsg.includes('429')) {
        displayMessage = '⏳ Too many requests. Please wait a moment and try again.';
      } else if (errorMsg.includes('504') || errorMsg.toLowerCase().includes('timeout') || errorMsg.toLowerCase().includes('timed out')) {
        displayMessage = '⏱️ The response took too long to generate. This can happen with web search or deep research. Please try again — the server may have been busy.';
      } else if (errorMsg.includes('500') || errorMsg.includes('503')) {
        displayMessage = '⚠️ The server is temporarily unavailable. Please try again in a few moments.';
      } else if (errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError')) {
        displayMessage = '🌐 Unable to connect to the server. Please check your internet connection.';
      }

      const lastMsg = store.messages[store.messages.length - 1];
      if (lastMsg) {
        store.updateMessage(lastMsg.id, {
          content: displayMessage,
          isError: true,
          isStreaming: false,
        });
      }
    } finally {
      currentAbortController = null;
      store.setLoading(false);
      store.setStreaming(false);
    }
  },

  regenerateLastResponse: async () => {
    const store = useChatStore.getState();
    const messages = store.messages;

    const lastUserMsgIndex = [...messages].reverse().findIndex((m) => m.role === 'user');
    if (lastUserMsgIndex === -1) return;

    const lastUserMsg = messages[messages.length - 1 - lastUserMsgIndex];

    const lastMsg = messages[messages.length - 1];
    if (lastMsg.role === 'assistant') {
      store.removeMessage(lastMsg.id);
    }

    await chatActions.sendMessage(lastUserMsg.content);
  },
};
