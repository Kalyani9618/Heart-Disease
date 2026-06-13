
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

// ============================================================================
// Message Types (consolidated)
// ============================================================================

export interface Citation {
  id: string;
  source: string;
  content: string;
  relevance?: number;
}

export interface ToolExecution {
  tool: string;
  input: Record<string, unknown>;
  output: unknown;
  status: 'success' | 'error';
  timestamp?: string;
}

/**
 * Unified Message interface for chat interactions
 * Compatible with both frontend display and backend API
 */
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;

  // UI & Extended fields
  type?: 'text' | 'action_request' | 'action_result' | 'widget';
  actionData?: any;
  widgetData?: {
    type: string;
    title: string;
    data: any;
  };
  image?: string; // Base64

  isError?: boolean;
  isStreaming?: boolean;
  thinkingContent?: string; // Alias for thinkingProcess?
  thinkingProcess?: string;

  // RAG & Grounding
  ragContext?: boolean;
  sources?: Array<{
    title: string;
    category?: string;
    relevance?: number;
  }>;
  groundingMetadata?: {
    groundingChunks?: Array<{
      web?: {
        uri?: string;
        title?: string;
      };
    }>;
  };

  citations?: Citation[];
  toolExecutions?: ToolExecution[];
  metadata?: Record<string, unknown> & {
    model?: string;
    processingTime?: number;
    tokens?: number;
    memoryContext?: string[];
    citations?: Citation[];
    is_emergency?: boolean;
  };

  // Agent routing fields (used by multi-agent / doctor mode)
  agentType?: 'doctor' | 'general' | string;
  routingReason?: string;
  complexityScore?: number;
  structuredData?: StructuredMedicalData;
}

/**
 * @deprecated Use Message with role: 'user' | 'assistant' instead
 * Kept for backward compatibility with older components
 */
export interface LegacyMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: Date;
  groundingMetadata?: unknown;
}

// ============================================================================
// Health Types
// ============================================================================

export enum HealthStatus {
  LowRisk = 'Low Risk',
  ModerateRisk = 'Moderate Risk',
  HighRisk = 'High Risk'
}

export type RiskLevel = 'Low Risk' | 'Moderate Risk' | 'High Risk';

export interface VitalsData {
  // Blood pressure
  systolic?: number;
  diastolic?: number;

  // Cholesterol
  cholesterol?: number;
  ldl?: number;
  hdl?: number;
  triglycerides?: number;

  // Other vitals
  bloodGlucose?: number;
  heartRate?: number;
  weight?: number;
  bmi?: number;
  oxygenSaturation?: number;
  temperature?: number;
}

export interface HealthAssessment {
  date: string;
  score: number;
  risk: RiskLevel;
  details: string;
  vitals: VitalsData;
}

// ============================================================================
// Notification Types
// ============================================================================

export type NotificationType = 'info' | 'warning' | 'error' | 'success' | 'medication' | 'appointment';

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: NotificationType;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
  data?: Record<string, unknown>;
}

export interface Appointment {
  id: string;
  doctorName: string;
  specialty: string;
  date: string;
  time: string;
  avatar?: string;
  rating?: number;
  type: 'in-person' | 'video';
  location: string;
  summary?: string; // AI generated summary
}

export interface Provider {
  id: string;
  name: string;
  specialty: string;
  qualifications: string;
  rating: number;
  reviewCount: number;
  photoUrl: string;
  clinicName: string;
  address: string;
  languages: string[];
  telehealthAvailable: boolean;
  acceptedInsurances: string[];
  bio: string;
  experienceYears: number;
  acceptsNewPatients: boolean;
}

export interface Device {
  id: string;
  name: string;
  type: 'watch' | 'ring' | 'chest-strap';
  lastSync: string;
  status: 'connected' | 'disconnected';
  battery?: number;
}

export interface Medication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  times: string[]; // ["08:00", "20:00"]
  takenToday: boolean[]; // [true, false] corresponding to times
  instructions?: string;
  quantity?: number; // Pills remaining
  refillThreshold?: number; // When to warn
}

export interface Badge {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  unlocked: boolean;
  dateUnlocked?: string;
  progress?: number; // percentage 0-100
}

export interface FamilyMember {
  id: string;
  name: string;
  relation: string;
  avatar: string;
  accessLevel: 'read-only' | 'admin';
  status: 'Critical' | 'Warning' | 'Stable';
  lastActive: string;
}

// ============================================================================
// Structured Medical Data Types
// ============================================================================

export interface MedicationData {
  name: string;
  dosage_value?: number;
  dosage_unit?: string;
  frequency?: string;
  confidence?: number;
}

export interface LabValueData {
  test_name: string;
  value: string | number;
  unit: string;
  reference_min?: number;
  reference_max?: number;
  is_abnormal?: boolean;
}

export interface VitalsSummaryData {
  heart_rate?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  spo2?: number;
  status?: string;
  summary_text?: string;
}

export interface StructuredMedicalData {
  medications?: MedicationData[];
  lab_values?: LabValueData[];
  diagnoses?: string[];
  vitals_summary?: VitalsSummaryData;
  rag_sources?: string[];
}
