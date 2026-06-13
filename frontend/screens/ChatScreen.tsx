import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { apiClient } from '../services/apiClient';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Message } from '../types';
import { memoryService } from '../services/memoryService';
import { ChatMessageMarkdown } from '../components/MarkdownRenderer';
import { EnhancedChatMessage } from '../components/EnhancedChatMessage';
import { useChatStore, ChatSession, chatActions, groupSessionsByDate, ChatSettings, SearchMode } from '../store/useChatStore';
import { CameraCapture } from '../components/CameraCapture';
import { useAuth } from '../hooks/useAuth';
import { useOfflineStatus } from '../hooks/useOfflineStatus';
import { useProvider } from '../contexts/ProviderContext';
import { useToast } from '../components/Toast';

// Audio Helpers
function base64ToUint8Array(base64: string) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]);
      } else {
        reject(new Error('Failed to convert blob to base64'));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

type SidebarPanel = 'history' | 'settings' | null;

function useAutoResize(value: string) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }
  }, [value]);
  return ref;
}

function formatTime(ts: string) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
}

const searchModeLabel: Record<SearchMode, { icon: string; label: string; color: string }> = {
  default: { icon: '', label: '', color: '' },
  web_search: { icon: 'travel_explore', label: 'Web Search', color: 'text-blue-500 bg-blue-500/10 border-blue-200/40 dark:border-blue-500/20' },
  deep_search: { icon: 'psychology', label: 'Deep Analysis', color: 'text-purple-500 bg-purple-500/10 border-purple-200/40 dark:border-purple-500/20' },
  memory: { icon: 'history', label: 'Memory', color: 'text-amber-500 bg-amber-500/10 border-amber-200/40 dark:border-amber-500/20' },
};

const ChatScreen: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { showToast } = useToast();
  const { isOnline } = useOfflineStatus();
  const location = useLocation();

  const [input, setInput] = useState('');
  const [attachment, setAttachment] = useState<string | null>(null);
  const [documentFiles, setDocumentFiles] = useState<Array<{ file: File; name: string; type: string; size: number }>>([]);
  const [showCamera, setShowCamera] = useState(false);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [isUploadingDocs, setIsUploadingDocs] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useAutoResize(input);

  const [sidebarPanel, setSidebarPanel] = useState<SidebarPanel>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [contextMenuSession, setContextMenuSession] = useState<string | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingId, setIsPlayingId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);

  const [regeneratingId, setRegeneratingId] = useState<string | null>(null);

  // Search mode: what tools/context to use for the next message
  const [searchMode, setSearchMode] = useState<SearchMode>('default');
  const [activeSearchMode, setActiveSearchMode] = useState<SearchMode>('default');

  // Editing a user message
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);

  const {
    messages, sessions, currentSessionId,
    isLoading, isStreaming, isSearchingMemories,
    selectedModel, settings,
    createSession, loadSession, deleteSession,
    loadSessions, updateSessionTitle, setSelectedModel,
    setMessages, pinSession, archiveSession, deleteAllSessions,
    updateSettings,
  } = useChatStore();

  useEffect(() => {
    if (!user) return;
    loadSessions(user.id);
    if (!currentSessionId && sessions.length === 0) createSession();
  }, [user]);

  useEffect(() => {
    memoryService.syncContext();
    if (location.state?.autoSend) {
      handleSend(location.state.autoSend);
      window.history.replaceState({}, document.title);
    }
    return () => { audioContextRef.current?.close(); };
  }, [location.state]);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, isLoading, attachment, scrollToBottom]);

  const activeSessions = useMemo(() => sessions.filter(s => !s.isArchived), [sessions]);
  const pinnedSessions = useMemo(() => activeSessions.filter(s => s.isPinned), [activeSessions]);
  const unpinnedSessions = useMemo(() => activeSessions.filter(s => !s.isPinned), [activeSessions]);
  const groupedSessions = useMemo(() => groupSessionsByDate(unpinnedSessions), [unpinnedSessions]);
  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return null;
    const q = searchQuery.toLowerCase();
    return activeSessions.filter(s =>
      s.title.toLowerCase().includes(q) ||
      s.lastMessage?.toLowerCase().includes(q) ||
      s.messages?.some(m => m.content.toLowerCase().includes(q))
    );
  }, [searchQuery, activeSessions]);
  const archivedSessions = useMemo(() => sessions.filter(s => s.isArchived), [sessions]);
  const [showArchived, setShowArchived] = useState(false);
  const dateGroupOrder = ['Today', 'Yesterday', 'Previous 7 Days', 'Previous 30 Days', 'Older'];

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setAttachment(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleDocumentSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowed = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/csv', 'image/jpeg', 'image/png'];
    const newDocs: typeof documentFiles = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      if (f.size > maxSize) {
        showToast(`${f.name} is too large (max 10MB).`, 'error');
        continue;
      }
      if (!allowed.includes(f.type) && !f.name.match(/\.(pdf|doc|docx|txt|csv|jpg|jpeg|png)$/i)) {
        showToast(`${f.name}: unsupported file type.`, 'error');
        continue;
      }
      newDocs.push({ file: f, name: f.name, type: f.type, size: f.size });
    }
    if (newDocs.length > 0) setDocumentFiles(prev => [...prev, ...newDocs]);
    if (docInputRef.current) docInputRef.current.value = '';
  };

  const removeDocument = (index: number) => {
    setDocumentFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleCameraCapture = (imageSrc: string) => {
    setAttachment(imageSrc);
    setShowCamera(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getFileIcon = (type: string, name: string) => {
    if (type === 'application/pdf' || name.endsWith('.pdf')) return 'picture_as_pdf';
    if (type.includes('word') || name.match(/\.docx?$/)) return 'description';
    if (type === 'text/plain' || name.endsWith('.txt')) return 'article';
    if (type === 'text/csv' || name.endsWith('.csv')) return 'table_chart';
    if (type.startsWith('image/')) return 'image';
    return 'attach_file';
  };

  const getFileColor = (type: string, name: string) => {
    if (type === 'application/pdf' || name.endsWith('.pdf')) return 'text-red-500 bg-red-50 dark:bg-red-900/20';
    if (type.includes('word') || name.match(/\.docx?$/)) return 'text-blue-500 bg-blue-50 dark:bg-blue-900/20';
    if (type === 'text/plain' || name.endsWith('.txt')) return 'text-slate-500 bg-slate-50 dark:bg-slate-800';
    if (type === 'text/csv' || name.endsWith('.csv')) return 'text-green-500 bg-green-50 dark:bg-green-900/20';
    if (type.startsWith('image/')) return 'text-purple-500 bg-purple-50 dark:bg-purple-900/20';
    return 'text-slate-400 bg-slate-50 dark:bg-slate-800';
  };

  const removeAttachment = () => {
    setAttachment(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSend = async (textInput?: string) => {
    const text = textInput || input;
    const hasDocs = documentFiles.length > 0;
    if ((!text.trim() && !attachment && !hasDocs) || isLoading) return;
    if (!isOnline) { showToast('You are currently offline.', 'warning'); return; }
    const currentSearchMode = searchMode;

    // Build message content with document context
    let messageContent = text;
    if (hasDocs) {
      const docNames = documentFiles.map(d => d.name).join(', ');
      const docSuffix = `\n\n📎 Attached documents: ${docNames}`;
      messageContent = (text.trim() ? text : 'Please analyze the attached document(s)') + docSuffix;
    }
    if (attachment && !text.trim() && !hasDocs) {
      messageContent = 'Please analyze this image.';
    }

    setInput('');
    setAttachment(null);
    setDocumentFiles([]);
    setActiveSearchMode(currentSearchMode);
    setSearchMode('default');
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (docInputRef.current) docInputRef.current.value = '';
    await chatActions.sendMessage(messageContent, selectedModel, currentSearchMode);
    setActiveSearchMode('default');
  };

  const startRecording = async () => {
    if (!isOnline) { showToast('Voice recording requires internet.', 'warning'); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recorder.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(blob);
        stream.getTracks().forEach(t => t.stop());
      };
      recorder.start();
      setIsRecording(true);
    } catch { showToast('Could not access microphone.', 'error'); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) { mediaRecorderRef.current.stop(); setIsRecording(false); }
  };

  const transcribeAudio = async (blob: Blob) => {
    try {
      const b64 = await blobToBase64(blob);
      const res = await apiClient.transcribeAudio(b64);
      if (res.success) setInput(res.text);
      else showToast('Transcription failed: ' + res.error, 'error');
    } catch { showToast('Failed to transcribe audio.', 'error'); }
  };

  const playTTS = async (text: string, messageId: string) => {
    if (activeSourceNodeRef.current && isPlayingId === messageId) {
      activeSourceNodeRef.current.stop();
      activeSourceNodeRef.current = null;
      setIsPlayingId(null);
      return;
    }
    if (activeSourceNodeRef.current) { activeSourceNodeRef.current.stop(); activeSourceNodeRef.current = null; }
    setIsPlayingId(messageId);
    try {
      const response = await apiClient.synthesizeSpeech(text);
      if (response.success && response.audio) {
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = ctx;
        const data = base64ToUint8Array(response.audio);
        const buf = await ctx.decodeAudioData(data.buffer);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.onended = () => { setIsPlayingId(null); activeSourceNodeRef.current = null; };
        src.start(0);
        activeSourceNodeRef.current = src;
      } else {
        const u = new SpeechSynthesisUtterance(text);
        u.onend = () => setIsPlayingId(null);
        window.speechSynthesis.speak(u);
      }
    } catch {
      setIsPlayingId(null);
      if ('speechSynthesis' in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.onend = () => setIsPlayingId(null);
        window.speechSynthesis.speak(u);
      }
    }
  };

  const copyMessage = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(id);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch {}
  };

  const handleRenameSession = async (sessionId: string) => {
    if (editTitle.trim()) await updateSessionTitle(sessionId, editTitle);
    setEditingSessionId(null);
  };

  const regenerateMessage = async (messageId: string) => {
    setRegeneratingId(messageId);
    try { await chatActions.regenerateLastResponse(); } finally { setRegeneratingId(null); }
  };

  // Edit a sent user message: load it into input and remove it + the assistant response that followed
  const editUserMessage = (messageId: string) => {
    const msgIndex = messages.findIndex(m => m.id === messageId);
    if (msgIndex === -1) return;
    const msg = messages[msgIndex];
    if (msg.role !== 'user') return;

    // Load content into input
    setInput(msg.content);
    setEditingMessageId(messageId);

    // Remove this user message and any assistant messages that follow it
    const store = useChatStore.getState();
    const idsToRemove: string[] = [messageId];
    for (let i = msgIndex + 1; i < messages.length; i++) {
      idsToRemove.push(messages[i].id);
    }
    idsToRemove.forEach(id => store.removeMessage(id));

    // Focus the textarea
    setTimeout(() => textareaRef.current?.focus(), 100);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  useEffect(() => {
    if (contextMenuSession) {
      const close = () => setContextMenuSession(null);
      const timer = setTimeout(() => document.addEventListener('click', close), 0);
      return () => { clearTimeout(timer); document.removeEventListener('click', close); };
    }
  }, [contextMenuSession]);

  // SettingsPanel
  const SettingsPanel = () => {
    const [localSettings, setLocalSettings] = useState<ChatSettings>({ ...settings });
    const handleSave = () => { updateSettings(localSettings); showToast('Settings saved', 'success'); };
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 dark:border-slate-700/50 flex-shrink-0">
          <div className="flex items-center gap-2">
            <button onClick={() => setSidebarPanel('history')} className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500">
              <span className="material-symbols-outlined text-lg">arrow_back</span>
            </button>
            <h2 className="font-semibold text-slate-900 dark:text-white text-[15px]">Settings</h2>
          </div>
          <button onClick={() => setSidebarPanel(null)} className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400">
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          <div>
            <label className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2.5 block">AI Model</label>
            <div className="grid grid-cols-1 gap-2">
              {(['medgemma'] as const).map(m => (
                <button key={m} onClick={() => setSelectedModel(m)}
                  className={`py-3 px-3 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                    selectedModel === m
                      ? 'bg-gradient-to-r from-red-500 to-red-600 text-white shadow-lg shadow-red-500/25'
                      : 'bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}>
                  <span className="material-symbols-outlined text-base">memory</span>
                  MedGemma
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2.5 block">Temperature</label>
            <input type="range" min="0" max="1" step="0.1" value={localSettings.temperature}
              onChange={e => setLocalSettings({ ...localSettings, temperature: parseFloat(e.target.value) })}
              className="w-full accent-red-500 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full cursor-pointer" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1.5"><span>Precise</span><span>Creative</span></div>
          </div>
          <div>
            <label className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2.5 block">System Prompt</label>
            <textarea rows={4} value={localSettings.systemPrompt}
              onChange={e => setLocalSettings({ ...localSettings, systemPrompt: e.target.value })}
              className="w-full bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-xl px-3.5 py-3 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500/40 resize-none transition-all" />
          </div>
          <div className="space-y-2.5">
            {[
              { key: 'streamResponses' as const, label: 'Stream Responses', desc: 'See tokens appear in real-time', icon: 'stream' },
              { key: 'autoGenerateTitle' as const, label: 'Auto-name Chats', desc: 'Generate title from first message', icon: 'title' },
            ].map(toggle => (
              <div key={toggle.key} className="flex items-center justify-between bg-slate-50 dark:bg-slate-800/60 rounded-xl px-4 py-3.5 border border-slate-100 dark:border-slate-700/30">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-base text-slate-400">{toggle.icon}</span>
                  <div>
                    <p className="text-sm text-slate-800 dark:text-white font-medium">{toggle.label}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">{toggle.desc}</p>
                  </div>
                </div>
                <button onClick={() => setLocalSettings({ ...localSettings, [toggle.key]: !localSettings[toggle.key] })}
                  className={`w-11 h-6 rounded-full transition-colors duration-200 relative flex-shrink-0 ${localSettings[toggle.key] ? 'bg-red-500' : 'bg-slate-300 dark:bg-slate-600'}`}>
                  <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${localSettings[toggle.key] ? 'left-[22px]' : 'left-0.5'}`} />
                </button>
              </div>
            ))}
          </div>
          <div className="pt-2">
            <label className="text-[11px] font-semibold text-red-400 uppercase tracking-wider mb-2.5 block">Danger Zone</label>
            <button onClick={() => { deleteAllSessions(); showToast('All chats deleted', 'success'); setSidebarPanel(null); }}
              className="w-full py-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm font-medium hover:bg-red-500/20 transition-all flex items-center justify-center gap-2">
              <span className="material-symbols-outlined text-base">delete_forever</span>
              Delete All Conversations
            </button>
          </div>
        </div>
        <div className="px-5 py-4 border-t border-slate-200 dark:border-slate-700/50 flex-shrink-0">
          <button onClick={handleSave}
            className="w-full py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl text-sm font-semibold hover:from-red-600 hover:to-red-700 transition-all shadow-lg shadow-red-500/25 active:scale-[0.98]">
            Save Settings
          </button>
        </div>
      </div>
    );
  };

  // SessionItem
  const SessionItem = ({ session }: { session: ChatSession }) => {
    const isActive = session.id === currentSessionId;
    const isEditing = editingSessionId === session.id;
    const showMenu = contextMenuSession === session.id;
    return (
      <div
        className={`sidebar-session-item group relative flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer text-sm ${
          isActive
            ? 'active bg-red-50/80 dark:bg-red-500/10 text-slate-900 dark:text-white'
            : 'text-slate-600 dark:text-slate-300'
        }`}
        onClick={() => { if (!isEditing) { loadSession(session.id); setSidebarPanel(null); } }}
      >
        <span className={`material-symbols-outlined text-base flex-shrink-0 ${isActive ? 'text-red-500' : 'text-slate-400'}`}>
          {session.isPinned ? 'push_pin' : 'chat_bubble_outline'}
        </span>
        {isEditing ? (
          <input type="text" value={editTitle}
            onChange={e => setEditTitle(e.target.value)}
            onBlur={() => handleRenameSession(session.id)}
            onKeyDown={e => { if (e.key === 'Enter') handleRenameSession(session.id); if (e.key === 'Escape') setEditingSessionId(null); }}
            autoFocus
            className="flex-1 bg-transparent border-b-2 border-red-500 text-sm focus:outline-none text-slate-900 dark:text-white min-w-0 py-0.5"
            onClick={e => e.stopPropagation()} />
        ) : (
          <div className="flex-1 min-w-0">
            <p className="text-[13px] truncate font-medium leading-tight">{session.title}</p>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 truncate mt-0.5 leading-tight">{session.lastMessage || `${session.messageCount} messages`}</p>
          </div>
        )}
        {!isEditing && (
          <button onClick={e => { e.stopPropagation(); setContextMenuSession(showMenu ? null : session.id); }}
            className="sm:opacity-0 sm:group-hover:opacity-100 p-1 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 transition-all flex-shrink-0">
            <span className="material-symbols-outlined text-base">more_horiz</span>
          </button>
        )}
        {showMenu && (
          <div className="absolute right-2 top-full mt-1 bg-white dark:bg-[#1a2737] border border-slate-200 dark:border-slate-700/60 rounded-xl shadow-xl z-[60] py-1 min-w-[160px] animate-scaleIn"
            onClick={e => e.stopPropagation()}>
            {[
              { icon: 'edit', label: 'Rename', action: () => { setEditingSessionId(session.id); setEditTitle(session.title); setContextMenuSession(null); } },
              { icon: session.isPinned ? 'push_pin' : 'push_pin', label: session.isPinned ? 'Unpin' : 'Pin', action: () => { pinSession(session.id); setContextMenuSession(null); } },
              { icon: 'archive', label: session.isArchived ? 'Unarchive' : 'Archive', action: () => { archiveSession(session.id); setContextMenuSession(null); } },
              { icon: 'delete', label: 'Delete', action: () => { deleteSession(session.id); setContextMenuSession(null); }, danger: true },
            ].map(item => (
              <button key={item.label} onClick={item.action}
                className={`w-full flex items-center gap-2.5 px-3.5 py-2 text-[12px] text-left transition-colors ${
                  (item as any).danger ? 'text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20' : 'text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/70'
                }`}>
                <span className="material-symbols-outlined text-sm">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Thinking stages simulation
  const [thinkingStage, setThinkingStage] = useState(0);
  useEffect(() => {
    if (!isLoading || messages.some(m => m.isStreaming)) { setThinkingStage(0); return; }
    const stages = activeSearchMode === 'web_search' ? 4 : activeSearchMode === 'deep_search' ? 5 : 3;
    const interval = setInterval(() => {
      setThinkingStage(prev => (prev + 1) % stages);
    }, 2200);
    return () => clearInterval(interval);
  }, [isLoading, activeSearchMode, messages]);

  const [thinkingExpanded, setThinkingExpanded] = useState(true);

  return (
    <div className="flex flex-col h-full bg-white dark:bg-[#0f1923] relative overflow-hidden">
      {/* SIDEBAR */}
      <AnimatePresence>
        {sidebarPanel !== null && (
          <>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/40 z-40 backdrop-blur-sm"
              onClick={() => setSidebarPanel(null)} />
            <motion.nav
              initial={{ x: -320 }} animate={{ x: 0 }} exit={{ x: -320 }}
              transition={{ type: 'spring', damping: 28, stiffness: 350 }}
              className="fixed top-0 left-0 h-screen w-[320px] max-w-[88vw] bg-white dark:bg-[#151f2b] z-50 shadow-2xl flex flex-col overflow-hidden">
              {sidebarPanel === 'settings' ? <SettingsPanel /> : (
                <>
                  {/* Sidebar Header */}
                  <div className="px-4 pt-5 pb-3 flex-shrink-0">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center">
                          <span className="material-symbols-outlined text-white text-sm">cardiology</span>
                        </div>
                        <h2 className="font-semibold text-slate-900 dark:text-white text-[15px]">Cardio AI</h2>
                      </div>
                      <button onClick={() => setSidebarPanel(null)} className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 transition-colors">
                        <span className="material-symbols-outlined text-lg">close</span>
                      </button>
                    </div>
                    <button onClick={() => { createSession(); setSidebarPanel(null); }}
                      className="w-full flex items-center gap-2.5 py-3 px-4 mb-3 border border-slate-200 dark:border-slate-700/50 text-slate-700 dark:text-slate-200 rounded-xl font-medium text-sm hover:bg-slate-50 dark:hover:bg-slate-800/60 active:scale-[0.98] transition-all">
                      <span className="material-symbols-outlined text-lg">add</span>
                      New Chat
                    </button>
                    <div className="relative">
                      <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-base">search</span>
                      <input type="text" placeholder="Search conversations..."
                        value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                        className="w-full bg-slate-50 dark:bg-[#0f1923] border border-slate-200 dark:border-slate-700/50 rounded-xl py-2.5 pl-10 pr-4 text-slate-700 dark:text-slate-200 text-[13px] focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500/30 placeholder:text-slate-400 transition-all" />
                      {searchQuery && (
                        <button onClick={() => setSearchQuery('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors">
                          <span className="material-symbols-outlined text-sm">close</span>
                        </button>
                      )}
                    </div>
                  </div>
                  {/* Session List */}
                  <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-0.5 chat-messages smooth-scroll">
                    {searchResults ? (
                      searchResults.length === 0 ? (
                        <div className="text-center py-12">
                          <span className="material-symbols-outlined text-3xl text-slate-300 dark:text-slate-600 mb-2 block">search_off</span>
                          <p className="text-xs text-slate-400">No results for "{searchQuery}"</p>
                        </div>
                      ) : (
                        <>
                          <p className="text-[10px] text-slate-400 uppercase tracking-wider px-2 py-2 font-semibold">
                            {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
                          </p>
                          {searchResults.map(s => <SessionItem key={s.id} session={s} />)}
                        </>
                      )
                    ) : (
                      <>
                        {pinnedSessions.length > 0 && (
                          <div className="mb-1">
                            <p className="text-[10px] text-slate-400 uppercase tracking-wider px-2 py-2 font-semibold flex items-center gap-1">
                              <span className="material-symbols-outlined text-[10px]">push_pin</span> Pinned
                            </p>
                            {pinnedSessions.map(s => <SessionItem key={s.id} session={s} />)}
                          </div>
                        )}
                        {dateGroupOrder.map(group =>
                          groupedSessions[group] ? (
                            <div key={group} className="mb-1">
                              <p className="text-[10px] text-slate-400 uppercase tracking-wider px-2 py-2 font-semibold">{group}</p>
                              {groupedSessions[group].map(s => <SessionItem key={s.id} session={s} />)}
                            </div>
                          ) : null
                        )}
                        {archivedSessions.length > 0 && (
                          <div className="mt-3">
                            <button onClick={() => setShowArchived(!showArchived)}
                              className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-wider px-2 py-2 font-semibold w-full hover:text-slate-300 transition-colors">
                              <span className="material-symbols-outlined text-xs transition-transform" style={{ transform: showArchived ? 'rotate(180deg)' : '' }}>expand_more</span>
                              Archived ({archivedSessions.length})
                            </button>
                            {showArchived && archivedSessions.map(s => <SessionItem key={s.id} session={s} />)}
                          </div>
                        )}
                        {activeSessions.length === 0 && (
                          <div className="text-center py-16">
                            <span className="material-symbols-outlined text-4xl text-slate-200 dark:text-slate-700 mb-3 block">forum</span>
                            <p className="text-sm text-slate-400 font-medium">No conversations yet</p>
                            <p className="text-[11px] text-slate-400/60 mt-1">Start a new chat to begin</p>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  <div className="px-3 py-3 border-t border-slate-100 dark:border-slate-700/30 flex-shrink-0 space-y-0.5">
                    {user && (
                      <div className="flex items-center gap-3 px-3 py-2.5 mb-1">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-400 to-slate-500 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                          {user.name?.charAt(0)?.toUpperCase() || 'U'}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-[13px] font-medium text-slate-700 dark:text-slate-200 truncate">{user.name || 'User'}</p>
                          <p className="text-[10px] text-slate-400 truncate">{user.email || ''}</p>
                        </div>
                      </div>
                    )}
                    <button onClick={() => setSidebarPanel('settings')}
                      className="w-full flex items-center gap-3 px-3 py-2.5 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/60 rounded-xl transition-colors text-sm">
                      <span className="material-symbols-outlined text-base text-slate-400">tune</span>
                      <span className="font-medium">Settings</span>
                    </button>
                  </div>
                </>
              )}
            </motion.nav>
          </>
        )}
      </AnimatePresence>

      {/* HEADER */}
      <header className="flex items-center justify-between px-3 py-2.5 z-10 bg-white/90 dark:bg-[#151f2b]/90 backdrop-blur-xl border-b border-slate-100 dark:border-slate-700/20 flex-shrink-0">
        <div className="flex items-center gap-1">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800/60 text-slate-500 dark:text-slate-400 active:scale-95 transition-all">
            <span className="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <button onClick={() => setSidebarPanel('history')} className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800/60 text-slate-500 dark:text-slate-400 active:scale-95 transition-all">
            <span className="material-symbols-outlined text-[20px]">menu</span>
          </button>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center">
            <span className="material-symbols-outlined text-white text-sm">cardiology</span>
          </div>
          <div className="flex flex-col">
            <h1 className="font-semibold text-[14px] text-slate-900 dark:text-white leading-tight">Cardio AI</h1>
            <div className="flex items-center gap-1.5">
              {isOnline ? (
                <span className="flex items-center gap-1 text-[10px] text-emerald-500 font-medium">
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-gentle-pulse" />
                  Online
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[10px] text-amber-500 font-medium">
                  <span className="w-1.5 h-1.5 bg-amber-500 rounded-full" />
                  Offline
                </span>
              )}
              <span className="text-slate-300 dark:text-slate-600 text-[8px]">{'\u2022'}</span>
              <span className="text-[10px] text-slate-400 capitalize">{selectedModel}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-0.5">
          <button onClick={() => createSession()} className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800/60 text-slate-500 dark:text-slate-400 active:scale-95 transition-all" title="New chat">
            <span className="material-symbols-outlined text-[20px]">edit_square</span>
          </button>
        </div>
      </header>

      {/* MESSAGES AREA */}
      <div className="flex-1 overflow-y-auto chat-messages smooth-scroll" style={{ minHeight: 0 }} role="log" aria-label="Chat messages" aria-live="polite">
        <div className="flex flex-col justify-end" style={{ minHeight: '100%' }}>
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-12 animate-fadeIn chat-content-width px-4" style={{ minHeight: '100%' }}>
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center shadow-lg shadow-red-500/15 mb-6 animate-breathe">
              <span className="material-symbols-outlined text-white text-3xl">cardiology</span>
            </div>
            <h2 className="text-2xl font-bold text-center mb-1.5">
              <span className="gradient-text">{getGreeting()}{user?.name?.trim() ? `, ${user.name.trim().split(' ')[0]}` : ''}</span>
            </h2>
            <p className="text-[15px] text-slate-500 dark:text-slate-400 text-center max-w-[340px] mb-10 leading-relaxed">
              How can I help you with your heart health today?
            </p>
            <div className="grid grid-cols-2 gap-3 w-full max-w-md">
              {[
                { icon: 'monitor_heart', label: 'Log Blood Pressure', desc: 'Record your latest reading', prompt: 'Log my blood pressure as 120 over 80 and heart rate 72', color: 'text-red-500', hoverBg: 'hover:bg-red-50 dark:hover:bg-red-500/5', border: 'border-slate-200 dark:border-slate-700/50 hover:border-red-200 dark:hover:border-red-500/30' },
                { icon: 'show_chart', label: 'View HR Trend', desc: 'Weekly heart rate data', prompt: 'Show my heart rate trend for this week', color: 'text-blue-500', hoverBg: 'hover:bg-blue-50 dark:hover:bg-blue-500/5', border: 'border-slate-200 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-500/30' },
                { icon: 'ecg_heart', label: 'Risk Assessment', desc: 'Check cardiovascular risk', prompt: 'Can you assess my cardiovascular risk based on my recent vitals?', color: 'text-emerald-500', hoverBg: 'hover:bg-emerald-50 dark:hover:bg-emerald-500/5', border: 'border-slate-200 dark:border-slate-700/50 hover:border-emerald-200 dark:hover:border-emerald-500/30' },
                { icon: 'medication', label: 'Medication Info', desc: 'Side effects & interactions', prompt: 'What are the side effects of Metoprolol?', color: 'text-purple-500', hoverBg: 'hover:bg-purple-50 dark:hover:bg-purple-500/5', border: 'border-slate-200 dark:border-slate-700/50 hover:border-purple-200 dark:hover:border-purple-500/30' },
              ].map((item, i) => (
                <button key={i} onClick={() => handleSend(item.prompt)}
                  style={{ animationDelay: `${i * 70}ms` }}
                  className={`flex flex-col items-start gap-1.5 p-4 rounded-2xl border transition-all duration-200 hover:shadow-md active:scale-[0.97] animate-floatUp text-left ${item.hoverBg} ${item.border}`}>
                  <span className={`material-symbols-outlined text-xl ${item.color}`}>{item.icon}</span>
                  <div>
                    <span className="text-[13px] font-medium text-slate-800 dark:text-slate-200 block leading-tight">{item.label}</span>
                    <span className="text-[11px] text-slate-400 dark:text-slate-500 block mt-0.5">{item.desc}</span>
                  </div>
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1.5 mt-10 px-3 py-2 bg-slate-50 dark:bg-slate-800/30 rounded-full">
              <span className="material-symbols-outlined text-[11px] text-amber-500">shield</span>
              <span className="text-[10px] text-slate-400">Not a substitute for professional medical advice</span>
            </div>
          </div>
        )}

        <div className="space-y-0 py-3">
          <AnimatePresence initial={false}>
            {messages.map((msg, idx) => {
              const isUser = msg.role === 'user';
              const isLast = idx === messages.length - 1;

              if (msg.type === 'action_request') {
                return (
                  <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                    className="flex justify-center my-2 chat-content-width px-4">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-50 dark:bg-amber-500/10 border border-amber-200/60 dark:border-amber-500/20 rounded-full">
                      <span className="material-symbols-outlined text-amber-500 text-sm animate-gentle-pulse">settings</span>
                      <span className="text-xs text-amber-700 dark:text-amber-300 italic">{msg.content}</span>
                    </div>
                  </motion.div>
                );
              }

              if (msg.type === 'action_result') {
                const { name } = msg.actionData;
                const config: Record<string, { icon: string; color: string; bg: string }> = {
                  logBiometrics: { icon: 'monitor_heart', color: 'text-red-500', bg: 'bg-red-50 dark:bg-red-500/10 border-red-200/60 dark:border-red-500/20' },
                  addMedication: { icon: 'pill', color: 'text-blue-500', bg: 'bg-blue-50 dark:bg-blue-500/10 border-blue-200/60 dark:border-blue-500/20' },
                  scheduleAppointment: { icon: 'calendar_today', color: 'text-purple-500', bg: 'bg-purple-50 dark:bg-purple-500/10 border-purple-200/60 dark:border-purple-500/20' },
                };
                const c = config[name] || { icon: 'check_circle', color: 'text-emerald-500', bg: 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200/60 dark:border-emerald-500/20' };
                return (
                  <motion.div key={msg.id} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    transition={{ type: 'spring', damping: 25 }}
                    className="flex justify-center my-2 chat-content-width px-4">
                    <div className={`flex items-center gap-3 px-5 py-3 rounded-2xl border ${c.bg}`}>
                      <span className={`material-symbols-outlined text-xl ${c.color}`}>{c.icon}</span>
                      <div>
                        <p className="text-sm font-semibold text-slate-800 dark:text-white">{msg.content.split(':')[0]}</p>
                        {msg.content.split(':')[1] && <p className="text-[11px] text-slate-500 mt-0.5">{msg.content.split(':')[1]}</p>}
                      </div>
                    </div>
                  </motion.div>
                );
              }

              if (msg.type === 'widget' && msg.widgetData) {
                const { type, title, data } = msg.widgetData;
                return (
                  <motion.div key={msg.id} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    transition={{ type: 'spring', damping: 25 }}
                    className="py-2 chat-content-width px-4">
                    <div className="w-full max-w-[400px] bg-white dark:bg-[#1a2737] rounded-2xl border border-slate-200 dark:border-slate-700/40 overflow-hidden shadow-sm">
                      <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-700/30 flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 uppercase tracking-wide">{title}</span>
                        <span className="material-symbols-outlined text-sm text-slate-400">show_chart</span>
                      </div>
                      <div className="p-4">
                        {type.includes('Chart') && (
                          <div className="h-36 w-full min-w-0 min-h-0">
                            <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0} debounce={50}>
                              <AreaChart data={data}>
                                <defs>
                                  <linearGradient id={`grad${msg.id}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2} />
                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                  </linearGradient>
                                </defs>
                                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }} itemStyle={{ color: '#fff' }} />
                                <Area type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill={`url(#grad${msg.id})`} />
                                <XAxis dataKey="day" hide />
                                <YAxis hide domain={['auto', 'auto']} />
                              </AreaChart>
                            </ResponsiveContainer>
                            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                              {data.length > 0 && <span>{data[0].day}</span>}
                              {data.length > 0 && <span>{data[data.length - 1].day}</span>}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                );
              }

              return (
                <motion.div key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: 'spring', damping: 25, stiffness: 350, mass: 0.8 }}
                  className={`message-row group py-4 ${isUser ? 'bg-transparent' : 'bg-slate-50/50 dark:bg-slate-800/20'}`}>
                  <div className="chat-content-width px-4">
                    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                      {/* Avatar */}
                      {!isUser ? (
                        <div className="shrink-0 mt-0.5">
                          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center">
                            <span className="material-symbols-outlined text-white text-sm">cardiology</span>
                          </div>
                        </div>
                      ) : (
                        <div className="shrink-0 mt-0.5">
                          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-slate-500 to-slate-600 flex items-center justify-center text-white text-xs font-bold">
                            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
                          </div>
                        </div>
                      )}

                      {/* Content */}
                      <div className={`flex flex-col min-w-0 ${isUser ? 'items-end flex-1' : 'items-start flex-1'}`}>
                        {/* Role label */}
                        <span className={`text-[11px] font-semibold mb-1 ${isUser ? 'text-slate-500 dark:text-slate-400' : 'text-red-500 dark:text-red-400'}`}>
                          {isUser ? 'You' : 'Cardio AI'}
                        </span>

                        {/* Message body */}
                        <div className={`relative text-sm leading-relaxed ${
                          isUser
                            ? 'max-w-[85%] px-4 py-3 bg-slate-100 dark:bg-slate-800/80 text-slate-800 dark:text-slate-100 rounded-2xl rounded-tr-md'
                            : 'w-full text-slate-800 dark:text-slate-100'
                        }`}>
                          {msg.image && (
                            <div className="mb-2.5 rounded-xl overflow-hidden">
                              <img src={msg.image} alt="Uploaded" className="max-w-full max-h-48 object-cover rounded-xl" />
                            </div>
                          )}

                          {/* Thinking process (ChatGPT-style collapsible) */}
                          {!isUser && (msg.thinkingContent || msg.thinkingProcess) && (
                            <div className="mb-3 thinking-container rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-700/40 overflow-hidden">
                              <button
                                onClick={() => setThinkingExpanded(!thinkingExpanded)}
                                className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-slate-100/50 dark:hover:bg-slate-700/30 transition-colors"
                              >
                                <span className="material-symbols-outlined text-sm text-amber-500 thinking-sparkle">auto_awesome</span>
                                <span className="text-[12px] font-medium text-slate-500 dark:text-slate-400">Thought process</span>
                                <span className={`material-symbols-outlined text-sm text-slate-400 transition-transform ml-auto ${thinkingExpanded ? 'rotate-180' : ''}`}>expand_more</span>
                              </button>
                              {thinkingExpanded && (
                                <div className="px-3 pb-3 text-[12px] text-slate-500 dark:text-slate-400 leading-relaxed border-t border-slate-200/40 dark:border-slate-700/30 pt-2">
                                  <ChatMessageMarkdown content={msg.thinkingContent || msg.thinkingProcess || ''} />
                                </div>
                              )}
                            </div>
                          )}

                          {!isUser ? (
                            <EnhancedChatMessage
                              content={msg.content}
                              sources={msg.sources}
                              isStreaming={msg.isStreaming}
                              isError={msg.isError}
                              metadata={msg.metadata}
                            />
                          ) : (
                            <span className="leading-relaxed whitespace-pre-wrap">{msg.content}</span>
                          )}
                          {msg.isStreaming && (
                            <span className="streaming-cursor" />
                          )}
                        </div>

                        {/* Meta info row */}
                        <div className={`flex items-center gap-1.5 mt-1.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                          <span className="text-[10px] text-slate-400/60 dark:text-slate-500/60">
                            {formatTime(msg.timestamp)}
                          </span>
                          {isUser && !isLoading && (
                            <button onClick={() => editUserMessage(msg.id)}
                              className="p-1 rounded-lg text-slate-300 dark:text-slate-600 hover:text-slate-500 dark:hover:text-slate-300 hover:bg-slate-200/50 dark:hover:bg-slate-800/50 transition-all duration-150 opacity-0 group-hover:opacity-100 focus:opacity-100"
                              title="Edit message">
                              <span className="material-symbols-outlined text-[13px]">edit</span>
                            </button>
                          )}
                          {msg.ragContext && !isUser && (
                            <span className="text-[9px] text-blue-500 bg-blue-500/10 px-1.5 py-0.5 rounded-full flex items-center gap-0.5 font-medium">
                              <span className="material-symbols-outlined text-[9px]">history</span> Memory
                            </span>
                          )}
                          {isUser && (msg as any).searchMode && (msg as any).searchMode !== 'default' && (() => {
                            const sm = searchModeLabel[(msg as any).searchMode as SearchMode];
                            return (
                              <span className={`text-[9px] px-1.5 py-0.5 rounded-full flex items-center gap-0.5 font-medium border ${sm.color}`}>
                                <span className="material-symbols-outlined text-[9px]">{sm.icon}</span> {sm.label}
                              </span>
                            );
                          })()}
                          {msg.metadata?.model && !isUser && (
                            <span className="text-[9px] text-slate-400 bg-slate-100 dark:bg-slate-800/50 px-1.5 py-0.5 rounded-full">
                              {String(msg.metadata.model)}
                            </span>
                          )}
                        </div>

                        {/* Action buttons (ChatGPT-style hover bar) */}
                        {!isUser && !msg.isStreaming && msg.content && (
                          <div className="message-actions-bar flex items-center gap-0.5 mt-1.5 -ml-1 bg-white dark:bg-slate-900/80 border border-slate-200/60 dark:border-slate-700/40 rounded-lg px-1 py-0.5 shadow-sm">
                            {[
                              { action: () => regenerateMessage(msg.id), icon: regeneratingId === msg.id ? 'sync' : 'refresh', title: 'Regenerate', active: regeneratingId === msg.id, activeClass: 'animate-spin text-blue-500' },
                              { action: () => copyMessage(msg.content, msg.id), icon: copiedMessageId === msg.id ? 'check' : 'content_copy', title: copiedMessageId === msg.id ? 'Copied!' : 'Copy', active: copiedMessageId === msg.id, activeClass: 'text-emerald-500' },
                              { action: () => playTTS(msg.content, msg.id), icon: isPlayingId === msg.id ? 'stop_circle' : 'volume_up', title: isPlayingId === msg.id ? 'Stop' : 'Read aloud', active: isPlayingId === msg.id, activeClass: 'text-red-500' },
                            ].map((btn, i) => (
                              <button key={i} onClick={btn.action} disabled={btn.icon === 'refresh' && isLoading}
                                className={`p-1.5 rounded-md transition-all duration-150 ${btn.active ? btn.activeClass : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/50'} disabled:opacity-30`}
                                title={btn.title}>
                                <span className="material-symbols-outlined text-[14px]">{btn.icon}</span>
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Grounding sources */}
                        {msg.groundingMetadata?.groundingChunks && (
                          <div className="flex flex-wrap gap-1.5 mt-2">
                            {msg.groundingMetadata.groundingChunks.map((chunk: any, i: number) => (
                              <a key={i} href={chunk.web?.uri || chunk.maps?.source?.uri} target="_blank" rel="noreferrer"
                                className="flex items-center gap-1 bg-blue-50 dark:bg-blue-500/10 border border-blue-200/50 dark:border-blue-500/20 px-2 py-1 rounded-lg text-[10px] text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-500/20 transition-colors">
                                <span className="material-symbols-outlined text-[10px]">link</span>
                                {chunk.web?.title || 'Source'}
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>

          {isLoading && !messages.some(m => m.isStreaming) && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              transition={{ type: 'spring', damping: 25 }}
              className="py-5 bg-slate-50/50 dark:bg-slate-800/20">
              <div className="chat-content-width px-4">
                <div className="flex items-start gap-3">
                  <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center flex-shrink-0 animate-breathe">
                    <span className="material-symbols-outlined text-white text-sm">cardiology</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="text-[11px] font-semibold text-red-500 dark:text-red-400 mb-2 block">Cardio AI</span>

                    {/* Thinking Container - ChatGPT/Gemini style */}
                    <div className="thinking-container rounded-xl bg-white dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/40 overflow-hidden">
                      {/* Thinking steps */}
                      <div className="p-3.5">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="material-symbols-outlined text-base text-amber-500 thinking-sparkle">auto_awesome</span>
                          <span className="text-[13px] font-medium text-slate-700 dark:text-slate-200">
                            {activeSearchMode === 'web_search' ? 'Searching the web...' : activeSearchMode === 'deep_search' ? 'Deep analysis...' : activeSearchMode === 'memory' ? 'Recalling your history...' : 'Thinking...'}
                          </span>
                          <button onClick={() => chatActions.stopGeneration()}
                            className="ml-auto p-1 rounded-md text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-all"
                            title="Stop generating">
                            <span className="material-symbols-outlined text-[14px]">stop_circle</span>
                          </button>
                        </div>

                        {/* Progress steps */}
                        <div className="space-y-2">
                          {(activeSearchMode === 'web_search'
                            ? [
                                { icon: 'manage_search', label: 'Understanding your question' },
                                { icon: 'travel_explore', label: 'Searching medical sources' },
                                { icon: 'fact_check', label: 'Verifying information' },
                                { icon: 'edit_note', label: 'Composing response' },
                              ]
                            : activeSearchMode === 'deep_search'
                            ? [
                                { icon: 'psychology', label: 'Analyzing query depth' },
                                { icon: 'biotech', label: 'Reviewing clinical literature' },
                                { icon: 'schema', label: 'Cross-referencing data' },
                                { icon: 'fact_check', label: 'Validating findings' },
                                { icon: 'edit_note', label: 'Synthesizing detailed response' },
                              ]
                            : [
                                { icon: 'manage_search', label: 'Understanding your question' },
                                { icon: 'neurology', label: 'Processing health context' },
                                { icon: 'edit_note', label: 'Generating response' },
                              ]
                          ).map((step, i) => (
                            <div key={i} className={`flex items-center gap-2.5 transition-all duration-500 ${
                              i < thinkingStage ? 'opacity-50' : i === thinkingStage ? 'opacity-100' : 'opacity-30'
                            }`}>
                              <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-500 ${
                                i < thinkingStage
                                  ? 'bg-emerald-100 dark:bg-emerald-500/20'
                                  : i === thinkingStage
                                  ? 'bg-red-100 dark:bg-red-500/20'
                                  : 'bg-slate-100 dark:bg-slate-700/50'
                              }`}>
                                {i < thinkingStage ? (
                                  <span className="material-symbols-outlined text-[11px] text-emerald-500">check</span>
                                ) : i === thinkingStage ? (
                                  <span className={`material-symbols-outlined text-[11px] text-red-500 thinking-step-dot`}>{step.icon}</span>
                                ) : (
                                  <span className="material-symbols-outlined text-[11px] text-slate-400">{step.icon}</span>
                                )}
                              </div>
                              <span className={`text-[12px] transition-colors duration-300 ${
                                i < thinkingStage
                                  ? 'text-slate-400 dark:text-slate-500 line-through'
                                  : i === thinkingStage
                                  ? 'text-slate-700 dark:text-slate-200 font-medium'
                                  : 'text-slate-400 dark:text-slate-500'
                              }`}>
                                {step.label}
                              </span>
                              {i === thinkingStage && (
                                <div className="flex gap-0.5 ml-1">
                                  {[0, 1, 2].map(d => (
                                    <div key={d} className="w-1 h-1 rounded-full bg-red-400" style={{ animation: 'thinkingDotWave 1.2s ease-in-out infinite', animationDelay: `${d * 0.15}s` }} />
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>

                        {/* Shimmer bar */}
                        <div className="mt-3 h-1 rounded-full overflow-hidden bg-slate-100 dark:bg-slate-700/50">
                          <div className="h-full thinking-shimmer-bar rounded-full" />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
        <div ref={messagesEndRef} className="h-1 shrink-0" />
        </div>
      </div>

      {/* INPUT AREA */}
      <div className="bg-white dark:bg-[#0f1923] border-t border-slate-100 dark:border-slate-700/20" style={{ flexShrink: 0 }}>
        <div className="chat-content-width px-3 pt-3" style={{ paddingBottom: 'max(12px, env(safe-area-inset-bottom))' }}>

        {/* Image attachment preview */}
        {attachment && (
          <div className="mb-2.5 bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-700/40 p-2.5 rounded-xl flex items-center gap-3 animate-scaleIn">
            <div className="w-14 h-14 rounded-lg bg-cover bg-center border border-slate-200 dark:border-slate-600/40 flex-shrink-0 shadow-sm" style={{ backgroundImage: `url('${attachment}')` }} />
            <div className="flex-1 min-w-0">
              <p className="text-[13px] text-slate-700 dark:text-slate-300 font-medium">Image attached</p>
              <p className="text-[11px] text-slate-400 mt-0.5">Ready to analyze</p>
            </div>
            <button onClick={removeAttachment} className="w-7 h-7 rounded-lg bg-slate-200/60 dark:bg-slate-700 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 flex items-center justify-center transition-colors flex-shrink-0">
              <span className="material-symbols-outlined text-sm">close</span>
            </button>
          </div>
        )}

        {/* Document attachments preview */}
        {documentFiles.length > 0 && (
          <div className="mb-2.5 flex flex-wrap gap-2 animate-scaleIn">
            {documentFiles.map((doc, idx) => (
              <div key={idx} className="flex items-center gap-2 bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-700/40 py-2 px-3 rounded-xl group">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getFileColor(doc.type, doc.name)}`}>
                  <span className="material-symbols-outlined text-[16px]">{getFileIcon(doc.type, doc.name)}</span>
                </div>
                <div className="min-w-0 max-w-[140px]">
                  <p className="text-[12px] text-slate-700 dark:text-slate-300 font-medium truncate">{doc.name}</p>
                  <p className="text-[10px] text-slate-400">{formatFileSize(doc.size)}</p>
                </div>
                <button onClick={() => removeDocument(idx)} className="w-5 h-5 rounded-md bg-slate-200/60 dark:bg-slate-700 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 flex items-center justify-center transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100">
                  <span className="material-symbols-outlined text-[12px]">close</span>
                </button>
              </div>
            ))}
          </div>
        )}

        {!isLoading && messages.length > 0 && messages.length < 4 && !attachment && (
          <div className="flex gap-2 mb-2.5 overflow-x-auto no-scrollbar pb-0.5">
            {[
              { label: '\u2764\uFE0F Log BP', prompt: 'Log my blood pressure as 120 over 80 and heart rate 72' },
              { label: '\uD83D\uDCC8 HR Trend', prompt: 'Show my heart rate trend for this week' },
              { label: '\uD83D\uDC8A Medications', prompt: 'What medications am I currently taking?' },
            ].map((item, i) => (
              <button key={i} onClick={() => handleSend(item.prompt)}
                className="flex-shrink-0 px-3.5 py-2 bg-white dark:bg-slate-800/60 text-slate-600 dark:text-slate-300 text-[12px] font-medium rounded-xl border border-slate-200 dark:border-slate-700/50 hover:bg-slate-50 dark:hover:bg-slate-700 hover:border-slate-300 dark:hover:border-slate-600 hover:text-slate-900 dark:hover:text-white active:scale-95 transition-all shadow-sm">
                {item.label}
              </button>
            ))}
          </div>
        )}

        {/* Modern input capsule */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700/50 bg-slate-50 dark:bg-[#1a2737] transition-all duration-200 input-capsule">
          {/* Textarea */}
          <div className="px-4 pt-3 pb-1">
            <textarea ref={textareaRef}
              value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyPress}
              placeholder={attachment ? 'Ask about this image...' : documentFiles.length > 0 ? 'Ask about the attached document(s)...' : 'Message Cardio AI...'}
              rows={1}
              className="w-full bg-transparent text-slate-800 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none text-[15px] resize-none max-h-[140px] leading-relaxed"
              autoComplete="off" />
          </div>

          {/* Bottom toolbar inside capsule */}
          <div className="flex items-center justify-between px-2 pb-2 pt-0.5">
            {/* Left: tool buttons */}
            <div className="flex items-center gap-0.5">
              <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileSelect} />
              <input type="file" ref={docInputRef} className="hidden" accept=".pdf,.doc,.docx,.txt,.csv" multiple onChange={handleDocumentSelect} />

              {/* Unified attach button with popup menu */}
              <div className="relative">
                <button onClick={() => setShowAttachMenu(v => !v)} title="Attach"
                  className={`p-2 rounded-xl transition-all duration-150 active:scale-90 ${
                    attachment || documentFiles.length > 0
                      ? 'text-emerald-500 bg-emerald-500/10'
                      : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-700/50'
                  }`}>
                  <span className="material-symbols-outlined text-[20px]" style={{ transition: 'transform 0.2s', transform: showAttachMenu ? 'rotate(45deg)' : 'none' }}>add</span>
                </button>

                {showAttachMenu && (
                  <>
                    {/* Backdrop */}
                    <div className="fixed inset-0 z-40" onClick={() => setShowAttachMenu(false)} />
                    {/* Menu */}
                    <div className="absolute bottom-full left-0 mb-2 z-50 bg-white dark:bg-[#1e2d3d] border border-slate-200 dark:border-slate-700/60 rounded-xl shadow-xl py-1.5 min-w-[180px] animate-scaleIn origin-bottom-left">
                      <button onClick={() => { setShowAttachMenu(false); setShowCamera(true); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                        <span className="material-symbols-outlined text-[18px] text-violet-500">photo_camera</span>
                        Camera
                      </button>
                      <button onClick={() => { setShowAttachMenu(false); fileInputRef.current?.click(); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                        <span className="material-symbols-outlined text-[18px] text-emerald-500">add_photo_alternate</span>
                        Photos & Images
                      </button>
                      <div className="h-px bg-slate-100 dark:bg-slate-700/50 my-1 mx-3" />
                      <button onClick={() => { setShowAttachMenu(false); docInputRef.current?.click(); }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-[13px] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                        <span className="material-symbols-outlined text-[18px] text-blue-500">attach_file</span>
                        Documents & PDFs
                      </button>
                    </div>
                  </>
                )}
              </div>

              {/* Voice */}
              <button onClick={() => (isRecording ? stopRecording() : startRecording())} title={isRecording ? 'Stop recording' : 'Voice input'}
                className={`p-2 rounded-xl transition-all duration-150 active:scale-90 ${
                  isRecording
                    ? 'text-red-500 bg-red-500/10 animate-gentle-pulse'
                    : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-700/50'
                }`}>
                <span className="material-symbols-outlined text-[20px]">{isRecording ? 'stop_circle' : 'mic'}</span>
              </button>

              {/* Separator */}
              <div className="w-px h-5 bg-slate-200 dark:bg-slate-700/50 mx-1" />

              {/* Search mode chips */}
              {([
                { mode: 'web_search' as SearchMode, icon: 'travel_explore', label: 'Search' },
                { mode: 'deep_search' as SearchMode, icon: 'psychology', label: 'Deep' },
                { mode: 'memory' as SearchMode, icon: 'history', label: 'Memory' },
              ]).map(chip => (
                <button key={chip.mode}
                  onClick={() => setSearchMode(searchMode === chip.mode ? 'default' : chip.mode)}
                  title={chip.label}
                  className={`flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-[11px] font-medium transition-all duration-200 active:scale-95 ${
                    searchMode === chip.mode
                      ? 'bg-red-500/10 dark:bg-red-500/20 text-red-600 dark:text-red-400 ring-1 ring-red-500/30'
                      : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-700/50'
                  }`}>
                  <span className="material-symbols-outlined text-[14px]">{chip.icon}</span>
                  <span className="hidden sm:inline">{chip.label}</span>
                </button>
              ))}
            </div>

            {/* Right: send or stop button */}
            {isLoading || isStreaming ? (
              <button onClick={() => chatActions.stopGeneration()}
                className="p-2 rounded-xl bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-700 dark:hover:bg-slate-200 transition-all duration-200 active:scale-90 shadow-sm"
                title="Stop generating">
                <span className="material-symbols-outlined text-[18px]">stop</span>
              </button>
            ) : (
              <button onClick={() => handleSend()} disabled={!input.trim() && !attachment && documentFiles.length === 0}
                className={`p-2 rounded-xl transition-all duration-200 ${
                  input.trim() || attachment || documentFiles.length > 0
                    ? 'bg-slate-900 dark:bg-white text-white dark:text-slate-900 shadow-sm hover:bg-slate-700 dark:hover:bg-slate-200 active:scale-90'
                    : 'bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500'
                } disabled:opacity-40 disabled:cursor-not-allowed`}>
                <span className="material-symbols-outlined text-[18px]">arrow_upward</span>
              </button>
            )}
          </div>
        </div>

        {isRecording && (
          <div className="flex items-center justify-center gap-3 mt-2.5 py-2 px-4 bg-red-50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-900/20 animate-fadeIn">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            <div className="flex items-center gap-0.5 h-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="w-0.5 bg-red-400 rounded-full waveform-bar" style={{ animationDelay: `${i * 0.1}s` }} />
              ))}
            </div>
            <span className="text-[11px] text-red-500 dark:text-red-400 font-medium">Recording...</span>
            <button onClick={stopRecording} className="ml-2 text-[11px] font-semibold text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 px-3 py-1 rounded-lg hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors">
              Stop & Transcribe
            </button>
          </div>
        )}

        {searchMode !== 'default' && (
          <div className="flex items-center justify-center gap-1.5 mt-2">
            <span className="material-symbols-outlined text-[12px] text-red-500">
              {searchMode === 'web_search' ? 'travel_explore' : searchMode === 'deep_search' ? 'psychology' : 'history'}
            </span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">
              {searchMode === 'web_search' && 'Web search enabled — responses include web results'}
              {searchMode === 'deep_search' && 'Deep analysis enabled — detailed, thorough responses'}
              {searchMode === 'memory' && 'Memory context — referencing your health history'}
            </span>
          </div>
        )}

        <p className="text-center text-[9px] text-slate-300 dark:text-slate-600 mt-2 select-none pb-0.5">
          AI assistant for guidance only — consult your doctor for medical decisions
        </p>
        </div>
      </div>

      {/* Camera Modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
};

export default ChatScreen;
