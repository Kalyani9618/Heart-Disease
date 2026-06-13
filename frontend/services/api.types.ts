/**
 * API Types for Document and Vision Services
 *
 * TypeScript interfaces matching backend API responses.
 * Healthcare-grade with strict typing for safety.
 */

// ============================================================================
// Document Processing Types
// ============================================================================

export interface DocumentUploadResponse {
    document_id: string;
    filename: string;
    file_size: number;
    content_type: string;
    status: 'uploaded' | 'processing' | 'processed' | 'failed';
    created_at: string;
}

export interface ExtractedEntity {
    type: string;
    value: string;
    confidence: number;
    start_offset?: number;
    end_offset?: number;
}

export interface DocumentProcessResponse {
    document_id: string;
    text: string;
    metadata: Record<string, unknown>;
    processor_used: string;
    entities: ExtractedEntity[];
    tables: Array<Record<string, unknown>>;
    confidence: number;
}

export interface ClassificationRequest {
    document_id: string;
    text?: string;
}

export interface ClassificationResult {
    document_id: string;
    document_type: string;
    category: string;
    confidence: number;
    subcategories: string[];
    suggested_schema: string;
}

export interface DocumentClassification {
    document_type: string;
    confidence: number;
    sub_type?: string;
    category?: string;
}

export interface DocumentDetails {
    document_id: string;
    filename: string;
    file_size: number;
    content_type: string;
    status: string;
    created_at: string;
    processed_at?: string;
    text?: string;
    entities?: ExtractedEntity[];
    classification?: DocumentClassification;
}

// ============================================================================
// Vision Analysis Types
// ============================================================================

export interface ECGAnalysisResponse {
    rhythm: string;
    heart_rate_bpm?: number;
    abnormalities: string[];
    confidence: number;
    recommendations: string[];
    requires_review: boolean;
    analysis_time_ms: number;
    disclaimer: string;
}

export type VisionImageType = 'ecg' | 'document' | 'auto';

export interface VisionAnalysisRequest {
    image_base64: string;
    image_type: VisionImageType;
    context?: string;
}

export interface VisionAnalysisResponse {
    image_type: string;
    analysis: Record<string, unknown>;
    confidence: number;
    processing_time_ms: number;
    timestamp: string;
}

export interface SupportedVisionType {
    type: string;
    description: string;
    formats: string[];
}

export interface SupportedTypesResponse {
    supported_types: SupportedVisionType[];
    max_file_size_mb: number;
    recommendations: Record<string, string>;
}

// ============================================================================
// Calendar Types (Phase 3)
// ============================================================================

export type CalendarProvider = 'google' | 'outlook';

export interface CalendarCredentialsRequest {
    provider: CalendarProvider;
    access_token: string;
    refresh_token?: string;
    expires_at?: string;
}

export interface CalendarSyncRequest {
    provider: CalendarProvider;
    days_ahead: number;
    include_reminders: boolean;
}

export interface CalendarEvent {
    id: string;
    title: string;
    start_time: string;
    end_time: string;
    location?: string;
    description?: string;
    calendar_id: string;
    provider: CalendarProvider;
    synced_at: string;
}

export interface ReminderRequest {
    event_id: string;
    reminder_type: 'medication' | 'appointment' | 'custom';
    remind_at: string;
    channel: 'push' | 'email' | 'whatsapp';
    message?: string;
}

// ============================================================================
// Notification Types (Phase 3)
// ============================================================================

export type NotificationPlatform = 'ios' | 'android' | 'web';

export interface DeviceRegistrationRequest {
    device_token: string;
    platform: NotificationPlatform;
}

export interface DeliveryChannel {
    channel: 'push' | 'email' | 'whatsapp' | 'sms';
    enabled: boolean;
    destination?: string;
}

export interface WeeklySummaryPreferences {
    enabled: boolean;
    delivery_channels: DeliveryChannel[];
    preferred_day: number; // 0-6 (Sunday-Saturday)
    preferred_time: string; // HH:MM format
    timezone: string;
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIErrorResponse {
    detail: string;
    status_code?: number;
    error_type?: string;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Parsed medication data from vision analysis
 */
export interface ParsedMedication {
    name: string;
    dosage?: string;
    frequency?: string;
    instructions?: string;
    quantity?: number;
    manufacturer?: string;
    ndc?: string; // National Drug Code
}

/**
 * Service health status
 */
export interface ServiceStatus {
    available: boolean;
    service_name: string;
    last_check?: string;
    error_message?: string;
}

// ============================================================================
// Heart Disease Types
// ============================================================================

export interface HeartDiseasePredictionRequest {
    age: number;
    sex: number;
    chest_pain_type: number;
    resting_bp_s: number;
    cholesterol: number;
    fasting_blood_sugar: number;
    resting_ecg: number;
    max_heart_rate: number;
    exercise_angina: number;
    oldpeak: number;
    st_slope: number;
}

export interface TestResultDetail {
    test_name: string;
    value: string;
    normal_range: string;
    status: 'Normal' | 'Abnormal' | 'Borderline' | 'Critical';
    risk_contribution: 'Low' | 'Moderate' | 'High';
    explanation: string;
}

export interface HeartDiseasePredictionResponse {
    prediction: number;
    probability: number;
    risk_level: string;
    confidence: number;
    message: string;
    test_results?: TestResultDetail[];
    clinical_interpretation?: string;
    triage_level?: string;
    triage_actions?: string[];
    guidelines_cited?: string[];
    is_grounded?: boolean;
    quality_score?: number;
    needs_medical_attention: boolean;
    processing_time_ms?: number;
    metadata?: Record<string, any>;
}

// ============================================================================
// Audio & Speech Types
// ============================================================================

export interface AudioTranscriptionResponse {
    success: boolean;
    text: string;
    error?: string;
}

export interface TextToSpeechResponse {
    success: boolean;
    audio: string; // Base64 encoded audio
    error?: string;
}
