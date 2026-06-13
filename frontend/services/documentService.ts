/**
 * Document Service
 *
 * Handles document upload, OCR processing, and entity extraction.
 * Follows modular service architecture per integration requirements.
 *
 * Healthcare-grade error handling:
 * - Network failures are caught and reported user-friendly
 * - No PHI or stack traces exposed to users
 * - All errors logged for developers
 */

import {
    DocumentUploadResponse,
    DocumentProcessResponse,
    DocumentDetails,
    ExtractedEntity,
    ClassificationResult,
    APIErrorResponse,
} from './api.types';
import { authService } from './authService';

// Use environment variable or fallback to localhost
const API_BASE_URL = (import.meta as any).env?.VITE_NLP_SERVICE_URL || 'http://localhost:5001';

// Configuration
const UPLOAD_TIMEOUT_MS = 60000; // 60 seconds for large files
const PROCESS_TIMEOUT_MS = 120000; // 2 minutes for OCR processing
const MAX_FILE_SIZE_MB = 10;

// ============================================================================
// Error Handling
// ============================================================================

export class DocumentServiceError extends Error {
    constructor(
        public code: string,
        public userMessage: string,
        public developerMessage: string,
        public statusCode?: number
    ) {
        super(userMessage);
        this.name = 'DocumentServiceError';
    }
}

/**
 * Convert API errors to user-friendly messages
 * Never exposes PHI, stack traces, or technical details
 */
function handleError(error: unknown, context: string): never {
    console.error(`[DocumentService] ${context}:`, error);

    if (error instanceof DocumentServiceError) {
        throw error;
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new DocumentServiceError(
            'NETWORK_ERROR',
            'Unable to connect. Please check your internet connection.',
            `Network error during ${context}: ${error.message}`
        );
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
        throw new DocumentServiceError(
            'TIMEOUT',
            'Request timed out. Please try again.',
            `Timeout during ${context}`
        );
    }

    // Handle fetch response errors
    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status: number }).status;
        const detail = (error as { detail?: string }).detail;

        switch (status) {
            case 400:
                throw new DocumentServiceError(
                    'INVALID_REQUEST',
                    'Invalid document format. Please check your file.',
                    detail || 'Bad request',
                    400
                );
            case 413:
                throw new DocumentServiceError(
                    'FILE_TOO_LARGE',
                    `File is too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`,
                    detail || 'Payload too large',
                    413
                );
            case 415:
                throw new DocumentServiceError(
                    'UNSUPPORTED_TYPE',
                    'File type not supported. Please use JPG, PNG, or PDF.',
                    detail || 'Unsupported media type',
                    415
                );
            case 503:
                throw new DocumentServiceError(
                    'SERVICE_UNAVAILABLE',
                    'Document processing is temporarily unavailable. Please try again later.',
                    detail || 'Service unavailable',
                    503
                );
            default:
                throw new DocumentServiceError(
                    'SERVER_ERROR',
                    'Something went wrong. Please try again.',
                    detail || `HTTP ${status}`,
                    status
                );
        }
    }

    throw new DocumentServiceError(
        'UNKNOWN_ERROR',
        'Something went wrong. Please try again.',
        error instanceof Error ? error.message : String(error)
    );
}

// ============================================================================
// API Methods
// ============================================================================

/**
 * Upload a document for processing
 *
 * @param file - The file to upload (image or PDF)
 * @param userId - User identifier for tracking
 * @returns Document upload response with document_id
 *
 * @throws DocumentServiceError with user-friendly message
 */
export async function uploadDocument(
    file: File,
    userId: string
): Promise<DocumentUploadResponse> {
    // Validate file size before upload
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
        throw new DocumentServiceError(
            'FILE_TOO_LARGE',
            `File is too large (${fileSizeMB.toFixed(1)}MB). Maximum size is ${MAX_FILE_SIZE_MB}MB.`,
            `File size ${fileSizeMB}MB exceeds limit`
        );
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
            method: 'POST',
            headers: {
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: formData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        handleError(error, 'uploadDocument');
    }
}

/**
 * Process an uploaded document (OCR + entity extraction)
 *
 * @param documentId - ID from uploadDocument response
 * @returns Processed document with text and entities
 */
export async function processDocument(
    documentId: string
): Promise<DocumentProcessResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), PROCESS_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({ document_id: documentId }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        handleError(error, 'processDocument');
    }
}

/**
 * Retrieve a processed document by ID
 *
 * @param documentId - Document identifier
 * @returns Document details including extracted text and entities
 */
export async function getDocument(documentId: string): Promise<DocumentDetails> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'getDocument');
    }
}

/**
 * Extract medical entities from text
 *
 * @param text - Text content to analyze
 * @param documentType - Optional document type for context
 * @returns Array of extracted entities with confidence scores
 */
export async function extractEntities(
    text: string,
    documentType?: string
): Promise<ExtractedEntity[]> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/extract`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({
                text,
                document_type: documentType,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        const result = await response.json();
        return result.entities || [];
    } catch (error) {
        handleError(error, 'extractEntities');
    }
}

/**
 * Classify a document
 *
 * @param documentId - Document identifier
 * @param userId - User identifier
 * @param text - Optional text to aid classification
 * @returns Classification result
 */
export async function classifyDocument(
    documentId: string,
    userId: string,
    text?: string
): Promise<ClassificationResult> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/classify?user_id=${userId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({
                document_id: documentId,
                text: text,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'classifyDocument');
    }
}

/**
 * List documents for a user
 */
export async function listDocuments(
    userId: string,
    skip: number = 0,
    limit: number = 50,
    status?: string
): Promise<DocumentDetails[]> {
    const params = new URLSearchParams({
        user_id: userId,
        skip: skip.toString(),
        limit: limit.toString()
    });

    if (status) {
        params.append('status', status);
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/list?${params}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'listDocuments');
    }
}

/**
 * Delete a document
 */
export async function deleteDocument(
    documentId: string,
    userId: string
): Promise<{ message: string }> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}?user_id=${userId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'deleteDocument');
    }
}

/**
 * Upload and process a document in one call
 * Convenience method that chains upload + process
 *
 * @param file - The file to upload and process
 * @param userId - User identifier
 * @param onProgress - Optional progress callback (0-100)
 * @returns Processed document with text and entities
 */
export async function uploadAndProcess(
    file: File,
    userId: string,
    onProgress?: (progress: number, stage: 'uploading' | 'processing') => void
): Promise<DocumentProcessResponse> {
    // Stage 1: Upload (0-50%)
    onProgress?.(0, 'uploading');
    const uploadResult = await uploadDocument(file, userId);
    onProgress?.(50, 'uploading');

    // Stage 2: Process (50-100%)
    onProgress?.(50, 'processing');
    const processResult = await processDocument(uploadResult.document_id);
    onProgress?.(100, 'processing');

    return processResult;
}

// ============================================================================
// Exported Service Object
// ============================================================================

export const documentService = {
    uploadDocument,
    processDocument,
    getDocument,
    extractEntities,
    classifyDocument,
    uploadAndProcess,
    listDocuments,
    deleteDocument,

    // Configuration exports for UI components
    MAX_FILE_SIZE_MB,
    SUPPORTED_TYPES: ['image/jpeg', 'image/png', 'image/tiff', 'application/pdf'],
};

export default documentService;
