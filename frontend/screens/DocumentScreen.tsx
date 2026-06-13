import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../services/apiClient';
import { useUserStore } from '../store/useUserStore';
import { useAuth } from '../hooks/useAuth';
import { DocumentDetails } from '../services/api.types';
import ScreenHeader from '../components/ScreenHeader';
import { useToast } from '../components/Toast';

// Use the shared type from api.types
type Document = DocumentDetails;

// Supported preview types
const PREVIEWABLE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/svg+xml', 'application/pdf', 'text/plain'];
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico', '.tiff', '.tif'];
const PDF_EXTENSIONS = ['.pdf'];
const TEXT_EXTENSIONS = ['.txt', '.md', '.csv', '.log'];

export default function DocumentScreen() {
    const navigate = useNavigate();
    const { user } = useAuth();
    const { showToast } = useToast();
    const [loading, setLoading] = useState(false);
    const [documents, setDocuments] = useState<Document[]>([]);
    const [showUploadModal, setShowUploadModal] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [previewDoc, setPreviewDoc] = useState<Document | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewType, setPreviewType] = useState<'image' | 'pdf' | 'text' | 'unsupported'>('unsupported');
    const [textContent, setTextContent] = useState<string>('');

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Filter documents by search query
    const filteredDocuments = documents.filter(doc => {
        if (!searchQuery.trim()) return true;
        const q = searchQuery.toLowerCase();
        return (
            doc.filename?.toLowerCase().includes(q) ||
            doc.classification?.document_type?.toLowerCase().includes(q) ||
            doc.classification?.category?.toLowerCase().includes(q) ||
            doc.status?.toLowerCase().includes(q)
        );
    });

    useEffect(() => {
        loadDocuments();
    }, [user]);

    const loadDocuments = async () => {
        if (!user) return;
        setLoading(true);
        try {
            const response = await apiClient.getDocuments();
            // Backend returns { documents: [], total: 0, ... } — extract the array
            const raw = Array.isArray(response) ? response : (response as any)?.documents ?? [];
            // Normalise field names: backend may send 'id' instead of 'document_id'
            // and 'uploaded_at' instead of 'created_at'
            const docs: Document[] = raw.map((d: any) => ({
                ...d,
                document_id: d.document_id || d.id || d.doc_id || '',
                created_at: d.created_at || d.uploaded_at || new Date().toISOString(),
                content_type: d.content_type || d.classification?.document_type || 'medical',
            }));
            setDocuments(docs);
        } catch (error) {
            console.error('Failed to load documents:', error);
            setDocuments([]);
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (!files || files.length === 0 || !user) return;

        setUploading(true);
        let successCount = 0;
        let failCount = 0;

        try {
            for (let i = 0; i < files.length; i++) {
                try {
                    await apiClient.uploadDocument(files[i]);
                    successCount++;
                } catch (error) {
                    console.error(`Upload failed for ${files[i].name}:`, error);
                    failCount++;
                }
            }

            await loadDocuments();
            setShowUploadModal(false);

            if (failCount === 0) {
                showToast(`${successCount} document${successCount > 1 ? 's' : ''} uploaded successfully`, 'success');
            } else {
                showToast(`${successCount} uploaded, ${failCount} failed`, 'warning');
            }
        } catch (error) {
            console.error('Upload failed:', error);
            showToast('Failed to upload documents', 'error');
        } finally {
            setUploading(false);
            // Reset the file input so the same file(s) can be selected again
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const getIconForType = (type: string) => {
        switch (type.toLowerCase()) {
            case 'lab report': return 'biotech';
            case 'prescription': return 'prescriptions';
            case 'imaging': return 'radiology';
            default: return 'description';
        }
    };

    const getPreviewType = (filename: string, contentType?: string): 'image' | 'pdf' | 'text' | 'unsupported' => {
        // First try content/mime type if available
        if (contentType) {
            if (contentType.startsWith('image/')) return 'image';
            if (contentType === 'application/pdf') return 'pdf';
            if (contentType.startsWith('text/')) return 'text';
        }
        // Fallback to extension-based detection
        const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
        if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
        if (PDF_EXTENSIONS.includes(ext)) return 'pdf';
        if (TEXT_EXTENSIONS.includes(ext)) return 'text';
        return 'unsupported';
    };

    const handleViewDocument = async (doc: Document) => {
        setPreviewDoc(doc);
        setPreviewLoading(true);
        setTextContent('');

        // Initial type guess from filename extension
        let type = getPreviewType(doc.filename || '', (doc as any).content_type);

        try {
            const response = await fetch(`/api/documents/${doc.document_id}/download`);
            if (!response.ok) throw new Error('Failed to load document');
            const blob = await response.blob();

            // If extension-based detection failed, try again with the actual blob MIME type
            if (type === 'unsupported' && blob.type) {
                type = getPreviewType(doc.filename || '', blob.type);
            }
            // Also try the response Content-Type header as a last resort
            if (type === 'unsupported') {
                const contentType = response.headers.get('content-type') || '';
                if (contentType) {
                    type = getPreviewType(doc.filename || '', contentType);
                }
            }

            setPreviewType(type);

            if (type === 'unsupported') {
                setPreviewLoading(false);
                return;
            }

            if (type === 'text') {
                const text = await blob.text();
                setTextContent(text);
            } else {
                const url = URL.createObjectURL(blob);
                setPreviewUrl(url);
            }
        } catch (err) {
            console.error('Failed to load preview:', err);
            showToast('Failed to load document preview', 'error');
            setPreviewType('unsupported');
        } finally {
            setPreviewLoading(false);
        }
    };

    const closePreview = () => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        setPreviewDoc(null);
        setPreviewUrl(null);
        setPreviewType('unsupported');
        setTextContent('');
    };

    const handleDownload = async (doc: Document) => {
        try {
            const response = await fetch(`/api/documents/${doc.document_id}/download`);
            if (!response.ok) throw new Error('Download failed');
            const blob = await response.blob();

            // Try Capacitor Filesystem for native Android download
            try {
                const { Filesystem, Directory } = await import('@capacitor/filesystem');
                const reader = new FileReader();
                const base64Data = await new Promise<string>((resolve, reject) => {
                    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
                await Filesystem.writeFile({
                    path: doc.filename || 'document',
                    data: base64Data,
                    directory: Directory.Documents,
                });
                showToast('Saved to Documents folder', 'success');
            } catch {
                // Fallback: browser download
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = doc.filename || 'document';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('Document downloaded', 'success');
            }
        } catch {
            showToast('Failed to download document', 'error');
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Medical Records"
                subtitle="Secure Document Storage"
                rightIcon="add"
                onRightAction={() => setShowUploadModal(true)}
            />

            <div className="max-w-4xl mx-auto p-4 space-y-6">

                {/* Search / Filter (Placeholder for now) */}
                <div className="relative">
                    <span className="material-symbols-outlined absolute left-4 top-3.5 text-slate-400">search</span>
                    <input
                        type="text"
                        placeholder="Search records..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-12 pr-4 py-3 bg-white dark:bg-card-dark rounded-xl border-none shadow-sm focus:ring-2 focus:ring-primary/20 dark:text-white"
                    />
                </div>

                {loading ? (
                    <div className="flex justify-center py-12">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                    </div>
                ) : documents.length === 0 ? (
                    <div className="text-center py-16">
                        <div className="w-20 h-20 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                            <span className="material-symbols-outlined text-4xl text-slate-300">folder_off</span>
                        </div>
                        <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">No Documents Yet</h3>
                        <p className="text-slate-500 dark:text-slate-400 text-sm max-w-xs mx-auto mb-6">Upload your lab reports, prescriptions, or imaging results to keep them organized.</p>
                        <button
                            onClick={() => setShowUploadModal(true)}
                            className="bg-primary text-white px-6 py-3 rounded-xl font-bold hover:bg-primary-dark transition-colors shadow-lg shadow-primary/20"
                        >
                            Upload First Document
                        </button>
                    </div>
                ) : (
                    <div className="grid gap-4">
                        {filteredDocuments.length === 0 ? (
                            <div className="text-center py-8">
                                <span className="material-symbols-outlined text-3xl text-slate-300 mb-2">search_off</span>
                                <p className="text-slate-500 dark:text-slate-400 text-sm">No documents matching "{searchQuery}"</p>
                            </div>
                        ) : filteredDocuments.map((doc) => (
                            <div key={doc.document_id} onClick={() => handleViewDocument(doc)} className="bg-white dark:bg-card-dark p-4 rounded-2xl border border-slate-100 dark:border-slate-800 shadow-sm flex items-center gap-4 hover:shadow-md transition-shadow group cursor-pointer">
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${doc.classification?.document_type === 'Lab Report' ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20' :
                                    doc.classification?.document_type === 'Prescription' ? 'bg-green-50 text-green-600 dark:bg-green-900/20' :
                                        'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/20'
                                    }`}>
                                    <span className="material-symbols-outlined">{getIconForType(doc.classification?.document_type || 'default')}</span>
                                </div>

                                <div className="flex-1 min-w-0">
                                    <h4 className="font-bold text-slate-900 dark:text-white truncate">{doc.filename}</h4>
                                    <div className="flex items-center gap-2 mt-1">
                                        <span className="text-xs font-medium px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500">
                                            {doc.classification?.document_type}
                                        </span>
                                        <span className="text-xs text-slate-400">
                                            {new Date(doc.created_at).toLocaleDateString()}
                                        </span>
                                    </div>
                                    {doc.classification?.category && (
                                        <p className="text-sm text-slate-500 dark:text-slate-400 mt-2 line-clamp-2">
                                            {doc.classification.category}
                                        </p>
                                    )}
                                </div>

                                <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                    <button onClick={() => handleViewDocument(doc)} className="p-2 text-slate-400 hover:text-primary hover:bg-slate-50 dark:hover:bg-slate-800 rounded-full transition-colors" title="View">
                                        <span className="material-symbols-outlined">visibility</span>
                                    </button>
                                    <button onClick={() => handleDownload(doc)} className="p-2 text-slate-400 hover:text-primary hover:bg-slate-50 dark:hover:bg-slate-800 rounded-full transition-colors" title="Download">
                                        <span className="material-symbols-outlined">download</span>
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Document Preview Modal */}
            {previewDoc && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm animate-in fade-in" onClick={closePreview}>
                    <div className="bg-white dark:bg-slate-900 rounded-3xl w-full max-w-4xl max-h-[90vh] shadow-2xl overflow-hidden flex flex-col animate-in zoom-in-95" onClick={e => e.stopPropagation()}>
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
                            <div className="flex items-center gap-3 min-w-0">
                                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
                                    previewDoc.classification?.document_type === 'Lab Report' ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20' :
                                    previewDoc.classification?.document_type === 'Prescription' ? 'bg-green-50 text-green-600 dark:bg-green-900/20' :
                                    'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/20'
                                }`}>
                                    <span className="material-symbols-outlined">{getIconForType(previewDoc.classification?.document_type || 'default')}</span>
                                </div>
                                <div className="min-w-0">
                                    <h3 className="font-bold text-slate-900 dark:text-white truncate">{previewDoc.filename}</h3>
                                    <p className="text-xs text-slate-500">{previewDoc.classification?.document_type} • {new Date(previewDoc.created_at).toLocaleDateString()}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <button onClick={() => handleDownload(previewDoc)} className="p-2 text-slate-500 hover:text-primary hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors" title="Download">
                                    <span className="material-symbols-outlined">download</span>
                                </button>
                                <button onClick={closePreview} className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors">
                                    <span className="material-symbols-outlined">close</span>
                                </button>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-auto p-4 bg-slate-50 dark:bg-slate-800/50">
                            {previewLoading ? (
                                <div className="flex items-center justify-center py-20">
                                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
                                </div>
                            ) : previewType === 'image' && previewUrl ? (
                                <div className="flex items-center justify-center min-h-[300px]">
                                    <img src={previewUrl} alt={previewDoc.filename} className="max-w-full max-h-[70vh] rounded-lg shadow-lg object-contain" />
                                </div>
                            ) : previewType === 'pdf' && previewUrl ? (
                                <iframe src={previewUrl} className="w-full h-[70vh] rounded-lg border-0" title={previewDoc.filename} />
                            ) : previewType === 'text' ? (
                                <pre className="whitespace-pre-wrap font-mono text-sm text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-700 overflow-auto max-h-[70vh]">
                                    {textContent}
                                </pre>
                            ) : (
                                <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                                    <span className="material-symbols-outlined text-5xl mb-4">preview_off</span>
                                    <p className="font-medium">Preview not available for this file type</p>
                                    <p className="text-sm mt-1">Click download to view the file</p>
                                    <button onClick={() => handleDownload(previewDoc)} className="mt-4 px-6 py-2 bg-primary text-white rounded-xl font-bold hover:bg-primary-dark transition-colors">
                                        Download File
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Upload Modal */}
            {showUploadModal && (
                <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 sm:p-6 bg-slate-900/60 backdrop-blur-sm animate-in fade-in" onClick={() => setShowUploadModal(false)}>
                    <div className="bg-white dark:bg-slate-900 rounded-3xl w-full max-w-sm p-6 shadow-2xl animate-in slide-in-from-bottom-10" onClick={e => e.stopPropagation()}>
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-6">Upload Document</h3>

                        <div
                            onClick={() => fileInputRef.current?.click()}
                            className="border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-2xl p-8 text-center hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors cursor-pointer"
                        >
                            {uploading ? (
                                <div className="py-4">
                                    <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                                    <p className="font-bold text-slate-900 dark:text-white">Uploading...</p>
                                </div>
                            ) : (
                                <>
                                    <span className="material-symbols-outlined text-4xl text-slate-400 mb-2">cloud_upload</span>
                                    <p className="font-bold text-slate-900 dark:text-white">Tap to Select Files</p>
                                    <p className="text-xs text-slate-500 mt-1">PDF, JPG, PNG, TXT, DOC up to 10MB each</p>
                                    <p className="text-[10px] text-slate-400 mt-0.5">You can select multiple files at once</p>
                                </>
                            )}
                        </div>
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            onChange={handleFileUpload}
                            accept=".pdf,.jpg,.jpeg,.png,.txt,.doc,.docx,.md"
                            multiple
                        />

                        <button
                            onClick={() => setShowUploadModal(false)}
                            className="w-full mt-6 py-3.5 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
