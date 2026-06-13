/**
 * Vision Service
 *
 * Handles AI vision analysis including ECG and document analysis.
 * Follows modular service architecture per integration requirements.
 *
 * Healthcare-grade error handling:
 * - Medical disclaimers always included in responses
 * - Confidence indicators for all analysis results
 * - No PHI or stack traces exposed to users
 * - All errors logged for developers
 */

import {
    ECGAnalysisResponse,
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    VisionImageType,
    SupportedTypesResponse,
    ParsedMedication,
} from './api.types';
import { authService } from './authService';

// Use environment variable or fallback to localhost
const API_BASE_URL = (import.meta as any).env?.VITE_NLP_SERVICE_URL || 'http://localhost:5001';

// Configuration
const ANALYSIS_TIMEOUT_MS = 60000; // 60 seconds for AI analysis
const MAX_FILE_SIZE_MB = 10;

// ============================================================================
// Error Handling
// ============================================================================

export class VisionServiceError extends Error {
    constructor(
        public code: string,
        public userMessage: string,
        public developerMessage: string,
        public statusCode?: number
    ) {
        super(userMessage);
        this.name = 'VisionServiceError';
    }
}

/**
 * Convert API errors to user-friendly messages
 * Never exposes PHI, stack traces, or technical details
 */
function handleError(error: unknown, context: string): never {
    console.error(`[VisionService] ${context}:`, error);

    if (error instanceof VisionServiceError) {
        throw error;
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new VisionServiceError(
            'NETWORK_ERROR',
            'Unable to connect. Please check your internet connection.',
            `Network error during ${context}: ${error.message}`
        );
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
        throw new VisionServiceError(
            'TIMEOUT',
            'Analysis is taking longer than expected. Please try again.',
            `Timeout during ${context}`
        );
    }

    // Handle fetch response errors
    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status: number }).status;
        const detail = (error as { detail?: string }).detail;

        switch (status) {
            case 400:
                throw new VisionServiceError(
                    'INVALID_IMAGE',
                    'Could not process this image. Please try a clearer photo.',
                    detail || 'Bad request',
                    400
                );
            case 413:
                throw new VisionServiceError(
                    'FILE_TOO_LARGE',
                    `Image is too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`,
                    detail || 'Payload too large',
                    413
                );
            case 415:
                throw new VisionServiceError(
                    'UNSUPPORTED_FORMAT',
                    'Image format not supported. Please use JPG or PNG.',
                    detail || 'Unsupported media type',
                    415
                );
            case 503:
                throw new VisionServiceError(
                    'SERVICE_UNAVAILABLE',
                    'Vision analysis is temporarily unavailable. Please try again later.',
                    detail || 'Service unavailable',
                    503
                );
            default:
                throw new VisionServiceError(
                    'SERVER_ERROR',
                    'Something went wrong. Please try again.',
                    detail || `HTTP ${status}`,
                    status
                );
        }
    }

    throw new VisionServiceError(
        'UNKNOWN_ERROR',
        'Something went wrong. Please try again.',
        error instanceof Error ? error.message : String(error)
    );
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert a File to base64 string
 */
export function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result as string;
            // Remove data URL prefix (e.g., "data:image/jpeg;base64,")
            const base64 = result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
}

/**
 * Parse medication information from vision analysis result
 * Extracts medication name, dosage, and other details from AI analysis
 */
export function parseMedicationFromVisionResult(
    result: VisionAnalysisResponse
): ParsedMedication {
    const analysis = result.analysis || {};

    // Try to extract medication info from various possible response structures
    const entities = (analysis.entities as Array<{ type: string; value: string }>) || [];
    const text = (analysis.text as string) || '';

    let name = '';
    let dosage = '';
    let frequency = '';
    let instructions = '';
    let quantity: number | undefined;

    // Look for medication entities
    for (const entity of entities) {
        switch (entity.type.toLowerCase()) {
            case 'medication':
            case 'drug':
            case 'medicine':
                if (!name) name = entity.value;
                break;
            case 'dosage':
            case 'dose':
            case 'strength':
                if (!dosage) dosage = entity.value;
                break;
            case 'frequency':
            case 'schedule':
                if (!frequency) frequency = entity.value;
                break;
            case 'instructions':
            case 'directions':
                if (!instructions) instructions = entity.value;
                break;
            case 'quantity':
            case 'count':
                if (!quantity) quantity = parseInt(entity.value);
                break;
        }
    }

    // Fallback: try to parse from raw text using patterns
    if (!name && text) {
        // Common medication name patterns (simplified)
        const nameMatch = text.match(/(?:medication|drug|rx):\s*([A-Za-z]+)/i);
        if (nameMatch) name = nameMatch[1];
    }

    if (!dosage && text) {
        // Common dosage patterns
        const dosageMatch = text.match(/(\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|tablets?|caps?|capsules?))/i);
        if (dosageMatch) dosage = dosageMatch[1];
    }

    return {
        name: name || 'Unknown Medication',
        dosage: dosage || undefined,
        frequency: frequency || undefined,
        instructions: instructions || undefined,
        quantity: quantity,
    };
}

// ============================================================================
// API Methods
// ============================================================================

/**
 * Analyze an ECG image
 *
 * @param file - ECG image file
 * @param patientContext - Optional patient context for better analysis
 * @returns ECG analysis with rhythm, abnormalities, and recommendations
 */
export async function analyzeECG(
    file: File,
    patientContext?: string
): Promise<ECGAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (patientContext) {
        formData.append('patient_context', patientContext);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/vision/ecg/analyze`, {
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
        handleError(error, 'analyzeECG');
    }
}

/**
 * Analyze an ECG from base64 encoded image data
 *
 * @param base64Image - Base64 encoded ECG image
 * @param patientContext - Optional patient context
 * @returns ECG analysis results
 */
export async function analyzeECGBase64(
    base64Image: string,
    patientContext?: string
): Promise<ECGAnalysisResponse> {
    const formData = new FormData();
    formData.append('image_base64', base64Image);
    if (patientContext) {
        formData.append('patient_context', patientContext);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/vision/ecg/analyze-base64`, {
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
        handleError(error, 'analyzeECGBase64');
    }
}

/**
 * Generic vision analysis endpoint
 * Automatically detects image type if not specified
 *
 * @param imageBase64 - Base64 encoded image
 * @param imageType - Type of analysis (ecg, food, document, auto)
 * @param context - Optional context for analysis
 * @returns Vision analysis results
 */
export async function analyzeVision(
    imageBase64: string,
    imageType: VisionImageType = 'auto',
    context?: string
): Promise<VisionAnalysisResponse> {
    const request: VisionAnalysisRequest = {
        image_base64: imageBase64,
        image_type: imageType,
        context,
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/vision/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify(request),
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
        handleError(error, 'analyzeVision');
    }
}

/**
 * Get list of supported vision analysis types
 *
 * @returns Supported types with descriptions and format requirements
 */
export async function getSupportedTypes(): Promise<SupportedTypesResponse> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/vision/supported-types`, {
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
        handleError(error, 'getSupportedTypes');
    }
}

// ============================================================================
// Exported Service Object
// ============================================================================

export const visionService = {
    // ECG Analysis
    analyzeECG,
    analyzeECGBase64,

    // Generic Vision
    analyzeVision,
    getSupportedTypes,

    // Utilities
    fileToBase64,
    parseMedicationFromVisionResult,

    // Configuration exports for UI components
    MAX_FILE_SIZE_MB,
    SUPPORTED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/bmp'],

    // Medical disclaimer (must be shown prominently for ECG analysis)
    ECG_DISCLAIMER:
        '⚠️ This AI analysis is for informational purposes only. ' +
        'ECG interpretation should be performed by a qualified healthcare provider. ' +
        'Seek immediate medical attention if experiencing chest pain or other cardiac symptoms.',
};

export default visionService;
