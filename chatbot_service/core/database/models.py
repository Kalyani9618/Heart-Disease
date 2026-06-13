"""
SQLAlchemy models for the Cardio AI application.

These models represent the database schema and are used by Alembic
for migration generation and management.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    email = Column(String(255))
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    blood_type = Column(String(5))
    weight_kg = Column(Float)
    height_cm = Column(Float)
    known_conditions = Column(JSON)
    medications = Column(JSON)
    allergies = Column(JSON)
    is_active = Column(Boolean, default=True)
    phone = Column(String(50))
    avatar = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_user_id', 'user_id'),
    )


class Device(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    device_type = Column(String(50))
    model = Column(String(100))
    last_sync = Column(TIMESTAMP)
    is_active = Column(Boolean, default=True)

    __table_args__ = (
        Index('idx_device_user', 'user_id'),
    )


class PatientRecord(Base):
    __tablename__ = 'patient_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    record_type = Column(String(100))
    data = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_patient_user', 'user_id'),
        Index('idx_patient_created', 'user_id', 'created_at'),
    )


class Vital(Base):
    __tablename__ = 'vitals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    device_id = Column(String(255))
    metric_type = Column(String(50))
    value = Column(Float)
    unit = Column(String(20))
    recorded_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_vitals_user', 'user_id'),
        Index('idx_vitals_recorded', 'user_id', 'recorded_at'),
        Index('idx_vitals_metric', 'metric_type'),
    )


class HealthAlert(Base):
    __tablename__ = 'health_alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    alert_type = Column(String(50))
    severity = Column(String(20))
    message = Column(Text)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    resolved_at = Column(TIMESTAMP, nullable=True)

    __table_args__ = (
        Index('idx_alerts_user', 'user_id'),
    )


class MedicalKnowledgeBase(Base):
    __tablename__ = 'medical_knowledge_base'

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text)
    content_type = Column(String(100))
    embedding = Column(Text)  # Using TEXT instead of BLOB for compatibility
    metadata_json = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_knowledge_type', 'content_type'),
    )


class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    started_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    ended_at = Column(TIMESTAMP, nullable=True)

    __table_args__ = (
        Index('idx_session_user', 'user_id'),
    )


class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False)  # Not using FK to avoid compatibility issues
    message_type = Column(String(20))  # 'user' or 'assistant'
    content = Column(Text)
    metadata_json = Column(JSON)
    timestamp = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_message_session', 'session_id'),
    )


class NotificationFailure(Base):
    __tablename__ = 'notification_failures'

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Notification details
    notification_type = Column(String(50), nullable=False)  # 'email', 'push', 'sms'
    recipient = Column(String(255), nullable=False)  # Email or phone number
    subject = Column(String(500))
    content = Column(Text)
    
    # Failure tracking
    original_attempt_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=5)
    next_retry_at = Column(TIMESTAMP)
    
    # Error details
    last_error = Column(Text)
    status = Column(String(20), default='pending')  # 'pending', 'retrying', 'failed', 'succeeded'
    
    # Metadata
    user_id = Column(String(255))
    metadata_json = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_status_next_retry', 'status', 'next_retry_at'),
        Index('idx_nf_user_id', 'user_id'),
        Index('idx_nf_created_at', 'created_at'),
    )


class RagFeedback(Base):
    __tablename__ = 'rag_feedback'

    feedback_id = Column(String(255), primary_key=True)
    query = Column(Text)
    response_preview = Column(Text)
    rating = Column(Integer)  # 1 = thumbs up, -1 = thumbs down, 0 = neutral
    user_id = Column(String(255))
    timestamp = Column(String(255))  # Stored as ISO format string
    citations_count = Column(Integer)
    context_sources = Column(JSON)
    user_comment = Column(Text)

    __table_args__ = (
        Index('idx_feedback_rating_ts', 'rating', 'timestamp'),
        Index('idx_feedback_user', 'user_id'),
    )


class Medication(Base):
    __tablename__ = 'medications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    drug_name = Column(String(255), nullable=False)
    dosage = Column(String(100))
    frequency = Column(String(100))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    schedule = Column(JSON, default=list)  # e.g. ['08:00', '20:00']
    notes = Column(Text)
    quantity = Column(Integer, default=30)
    instructions = Column(Text)
    taken_today = Column(JSON, default=list)  # e.g. [false, true]
    times = Column(JSON, default=lambda: ['08:00'])  # time slots
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_medication_user', 'user_id'),
    )


class Feedback(Base):
    """
    New feedback model for the decoupled storage interface.
    """
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    result_id = Column(String(255), nullable=False)
    response_preview = Column(String(500))
    feedback_type = Column(String(50), nullable=False)
    rating = Column(Integer)
    comment = Column(Text)
    metadata_json = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_fb_user_id', 'user_id'),
        Index('idx_fb_result_id', 'result_id'),
        Index('idx_fb_feedback_type', 'feedback_type'),
    )


class EmergencyContact(Base):
    """Emergency contact for a user."""
    __tablename__ = 'emergency_contacts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    relation = Column(String(100))
    phone = Column(String(50))
    is_primary = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_emergency_user', 'user_id'),
    )


class UserAppSettings(Base):
    """Application-level settings per user."""
    __tablename__ = 'user_app_settings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, nullable=False)
    notifications_all = Column(Boolean, default=True)
    notifications_meds = Column(Boolean, default=True)
    notifications_insights = Column(Boolean, default=False)
    units = Column(String(20), default='Metric')
    theme = Column(String(20), default='light')
    language = Column(String(10), default='en')
    settings_json = Column(JSON, default=dict)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_app_settings_user', 'user_id'),
    )


class FamilyMember(Base):
    """Family member / caretaker relationship."""
    __tablename__ = 'family_members'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    member_user_id = Column(String(255))
    name = Column(String(255), nullable=False)
    relation = Column(String(100))
    avatar = Column(Text)
    access_level = Column(String(50), default='read-only')
    status = Column(String(50), default='Stable')
    last_active = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_family_user', 'user_id'),
    )


class UserDevice(Base):
    __tablename__ = 'user_devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    device_id = Column(String(255), unique=True, nullable=False)
    device_type = Column(String(100))
    device_name = Column(String(255))
    firmware_version = Column(String(100))
    battery = Column(Integer, default=100)
    status = Column(String(50), default='connected')
    last_sync = Column(TIMESTAMP)
    registered_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    is_active = Column(Boolean, default=True)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_user_device_user', 'user_id'),
        Index('idx_user_device_id', 'device_id'),
    )


class DeviceTimeSeries(Base):
    __tablename__ = 'device_timeseries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    ts = Column(DateTime, nullable=False)
    idempotency_key = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_dt_device', 'device_id'),
        Index('idx_dt_metric', 'metric_type'),
        Index('idx_dt_ts', 'ts'),
        Index('idx_dt_device_ts', 'device_id', 'ts'),
    )


# ============================================================================
# Appointment System Models
# ============================================================================

class Provider(Base):
    """Healthcare provider / doctor record."""
    __tablename__ = 'providers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(50), unique=True, nullable=False)  # e.g. 'p_101'
    name = Column(String(255), nullable=False)
    specialty = Column(String(100), nullable=False)
    qualifications = Column(String(255))
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    photo_url = Column(String(500))
    clinic_name = Column(String(255))
    address = Column(Text)
    languages = Column(JSON)  # ["English", "Spanish"]
    telehealth_available = Column(Boolean, default=False)
    accepted_insurances = Column(JSON)  # ["Aetna", "BlueCross"]
    bio = Column(Text)
    experience_years = Column(Integer, default=0)
    accepts_new_patients = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_provider_id', 'provider_id'),
        Index('idx_provider_specialty', 'specialty'),
        Index('idx_provider_active', 'is_active'),
    )


class ProviderAvailability(Base):
    """Recurring or specific-date availability slots for a provider."""
    __tablename__ = 'provider_availability'

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(50), nullable=False)  # references providers.provider_id
    date = Column(String(10), nullable=False)  # 'YYYY-MM-DD'
    time_slot = Column(String(5), nullable=False)  # 'HH:MM'
    is_booked = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_avail_provider', 'provider_id'),
        Index('idx_avail_date', 'provider_id', 'date'),
        Index('idx_avail_booked', 'provider_id', 'date', 'is_booked'),
    )


class AppointmentRecord(Base):
    """Patient appointment booking record."""
    __tablename__ = 'appointments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    appointment_id = Column(String(100), unique=True, nullable=False)  # e.g. 'apt_1700000000'
    user_id = Column(String(255), nullable=False)
    provider_id = Column(String(50), nullable=False)  # references providers.provider_id

    # Provider snapshot (denormalized for quick display)
    doctor_name = Column(String(255), nullable=False)
    specialty = Column(String(100))
    doctor_rating = Column(Float)

    # Schedule
    date = Column(String(10), nullable=False)  # 'YYYY-MM-DD'
    time = Column(String(5), nullable=False)   # 'HH:MM'
    duration_minutes = Column(Integer, default=30)
    appointment_type = Column(String(20), nullable=False, default='in-person')  # 'in-person' | 'video'

    # Location
    location = Column(String(255))
    virtual_link = Column(String(500))

    # Clinical
    reason = Column(Text)  # from intake
    intake_summary = Column(Text)  # AI triage result
    consultation_summary = Column(Text)  # post-visit AI summary
    shared_chart_data = Column(JSON)  # vitals shared with doctor

    # Insurance
    insurance_provider = Column(String(255))
    insurance_member_id = Column(String(100))
    insurance_group_id = Column(String(100))

    # Status
    status = Column(String(20), nullable=False, default='scheduled')
    # 'scheduled' | 'confirmed' | 'completed' | 'cancelled' | 'no_show'
    cancellation_reason = Column(Text)

    # Cost
    estimated_cost = Column(Float, default=150.00)
    actual_cost = Column(Float)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_appt_id', 'appointment_id'),
        Index('idx_appt_user', 'user_id'),
        Index('idx_appt_provider', 'provider_id'),
        Index('idx_appt_date', 'date'),
        Index('idx_appt_status', 'status'),
        Index('idx_appt_user_date', 'user_id', 'date'),
    )


class InsuranceInfo(Base):
    """Stored insurance information for a user."""
    __tablename__ = 'insurance_info'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    insurance_provider = Column(String(255), nullable=False)
    member_id = Column(String(100), nullable=False)
    group_id = Column(String(100))
    plan_type = Column(String(50))  # HMO, PPO, etc.
    is_primary = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_insurance_user', 'user_id'),
    )


# =========================================================================
# Consent System
# =========================================================================

class UserConsent(Base):
    """User consent records for GDPR/HIPAA compliance."""
    __tablename__ = 'user_consents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    consent_type = Column(String(100), nullable=False)  # e.g. 'data_processing'
    granted = Column(Boolean, default=False, nullable=False)
    description = Column(Text)
    required = Column(Boolean, default=False)
    granted_at = Column(TIMESTAMP)
    revoked_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_consent_user', 'user_id'),
        Index('idx_consent_user_type', 'user_id', 'consent_type', unique=True),
    )


# =========================================================================
# Calendar System
# =========================================================================

class CalendarCredential(Base):
    """Calendar integration credentials per user."""
    __tablename__ = 'calendar_credentials'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, unique=True)
    provider = Column(String(50), default='google')
    access_token = Column(Text)
    refresh_token = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    __table_args__ = (
        Index('idx_calcred_user', 'user_id'),
    )


class CalendarEvent(Base):
    """Calendar events synced or created."""
    __tablename__ = 'calendar_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    start_time = Column(String(30), nullable=False)
    end_time = Column(String(30), nullable=False)
    location = Column(String(255))
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_calevent_user', 'user_id'),
        Index('idx_calevent_start', 'user_id', 'start_time'),
    )


class CalendarReminder(Base):
    """Health-related reminders."""
    __tablename__ = 'calendar_reminders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    reminder_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)
    appointment_id = Column(String(100))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    scheduled_for = Column(String(30), nullable=False)
    reminder_minutes_before = Column(Integer, default=30)
    status = Column(String(20), default='scheduled')  # scheduled, sent, dismissed
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_calreminder_user', 'user_id'),
        Index('idx_calreminder_status', 'status'),
    )


# =========================================================================
# Push Notification Devices
# =========================================================================

class PushDevice(Base):
    """Registered push notification devices."""
    __tablename__ = 'push_devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    device_token = Column(String(500), nullable=False)
    platform = Column(String(20), nullable=False)  # ios, android, web
    registered_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_pushdev_user', 'user_id'),
        Index('idx_pushdev_token', 'user_id', 'device_token', unique=True),
    )


# =========================================================================
# Content Verification (Compliance)
# =========================================================================

class ContentVerification(Base):
    """Content verification queue for HIPAA compliance."""
    __tablename__ = 'content_verifications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(100), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)
    submitted_by = Column(String(255))
    status = Column(String(20), default='pending')  # pending, verified, rejected
    reviewer_notes = Column(Text)
    submitted_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    reviewed_at = Column(TIMESTAMP)

    __table_args__ = (
        Index('idx_verify_status', 'status'),
        Index('idx_verify_item', 'item_id'),
    )


# =========================================================================
# Prediction History
# =========================================================================

class PredictionHistory(Base):
    """Heart disease prediction history for longitudinal tracking."""
    __tablename__ = 'prediction_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(255))
    input_data = Column(JSON)
    prediction = Column(Integer)  # 0 or 1
    probability = Column(Float)
    risk_level = Column(String(20))
    confidence = Column(Float)
    clinical_interpretation = Column(Text)
    processing_time_ms = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    __table_args__ = (
        Index('idx_pred_user', 'user_id'),
        Index('idx_pred_created', 'created_at'),
    )