
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Provider, Appointment } from '../types';
import { apiClient } from '../services/apiClient';
import { useToast } from '../components/Toast';
import { useConfirm } from '../components/ConfirmDialog';

// User ID — in production this comes from auth context
const USER_ID = localStorage.getItem('user_id') || 'user123';

// --- New Intake Modal Component ---
const IntakeModal = ({
    doctorName,
    onComplete,
    onCancel
}: {
    doctorName: string,
    onComplete: (reason: string, urgency: string, summary: string) => void,
    onCancel: () => void
}) => {
    const [symptoms, setSymptoms] = useState('');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [triageResult, setTriageResult] = useState<'safe' | 'emergency' | null>(null);

    const handleAnalyze = async () => {
        if (!symptoms.trim()) return;
        setIsAnalyzing(true);

        try {
            // Use backend triage API
            const result = await apiClient.analyzeIntake(symptoms);

            if (result.urgency === 'emergency') {
                setTriageResult('emergency');
            } else {
                onComplete(symptoms, result.urgency, result.summary);
            }
        } catch (error) {
            console.error("Triage Error", error);
            // Allow booking if AI fails, default to routine
            onComplete(symptoms, 'Routine', symptoms);
        } finally {
            setIsAnalyzing(false);
        }
    };

    if (triageResult === 'emergency') {
        return (
            <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-red-900/90 backdrop-blur-md animate-in zoom-in-95">
                <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-sm shadow-2xl text-center border-4 border-red-500">
                    <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
                        <span className="material-symbols-outlined text-4xl text-red-600">warning</span>
                    </div>
                    <h2 className="text-2xl font-bold text-red-600 mb-2">Medical Emergency</h2>
                    <p className="text-slate-700 dark:text-slate-300 mb-6 font-medium">
                        Your symptoms suggest a potentially life-threatening condition. Do not book an appointment.
                    </p>
                    <button onClick={async () => {
                        const telUrl = 'tel:911';
                        try {
                            const { Browser } = await import('@capacitor/browser');
                            await Browser.open({ url: telUrl });
                        } catch {
                            window.location.href = telUrl;
                        }
                    }} className="w-full py-4 bg-red-600 hover:bg-red-700 text-white font-bold rounded-xl shadow-lg shadow-red-500/40 flex items-center justify-center gap-2 mb-3">
                        <span className="material-symbols-outlined">call</span> Call Emergency Services
                    </button>
                    <button onClick={onCancel} className="text-sm text-slate-500 underline">Cancel Booking</button>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in">
            <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-sm shadow-2xl relative">
                <button onClick={onCancel} className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"><span className="material-symbols-outlined">close</span></button>
                <div className="mb-4">
                    <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mb-3 text-blue-600 dark:text-blue-400">
                        <span className="material-symbols-outlined">clinical_notes</span>
                    </div>
                    <h3 className="text-xl font-bold dark:text-white">Reason for Visit</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Tell Dr. {doctorName} why you are booking this appointment.</p>
                </div>

                <textarea
                    value={symptoms}
                    onChange={(e) => setSymptoms(e.target.value)}
                    placeholder="e.g. I've been feeling dizzy after workouts..."
                    className="w-full h-32 p-4 rounded-xl bg-slate-50 dark:bg-slate-800 border-none outline-none focus:ring-2 focus:ring-primary dark:text-white resize-none mb-4"
                ></textarea>

                <div className="bg-yellow-50 dark:bg-yellow-900/10 p-3 rounded-lg flex items-start gap-2 mb-4">
                    <span className="material-symbols-outlined text-yellow-600 text-sm mt-0.5">info</span>
                    <p className="text-xs text-yellow-700 dark:text-yellow-400">AI will analyze your input to ensure this isn't an emergency.</p>
                </div>

                <button
                    onClick={handleAnalyze}
                    disabled={!symptoms.trim() || isAnalyzing}
                    className="w-full py-3 bg-primary hover:bg-primary-dark text-white font-bold rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                >
                    {isAnalyzing ? (
                        <>
                            <span className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin"></span>
                            Analyzing...
                        </>
                    ) : (
                        <>
                            Continue to Scheduling <span className="material-symbols-outlined text-sm">arrow_forward</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};

const ReceptionistModal = ({ onClose }: { onClose: () => void }) => {
    const navigate = useNavigate();
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
            <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-sm shadow-2xl relative" onClick={e => e.stopPropagation()}>
                <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-slate-600">
                    <span className="material-symbols-outlined">close</span>
                </button>
                <div className="text-center mb-6">
                    <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-3 text-blue-600 dark:text-blue-400">
                        <span className="material-symbols-outlined text-3xl">support_agent</span>
                    </div>
                    <h3 className="text-xl font-bold dark:text-white">Reception Desk</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">How can we help you today?</p>
                </div>

                <div className="space-y-3">
                    <button
                        onClick={() => window.location.href = 'tel:+18001234567'}
                        className="w-full p-4 bg-slate-50 dark:bg-slate-800 rounded-xl flex items-center gap-4 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center text-green-600 dark:text-green-400">
                            <span className="material-symbols-outlined">call</span>
                        </div>
                        <div className="text-left">
                            <p className="font-bold text-sm dark:text-white">Call Us</p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">+1 (800) 123-4567</p>
                        </div>
                    </button>

                    <button
                        onClick={() => { onClose(); navigate('/chat'); }}
                        className="w-full p-4 bg-slate-50 dark:bg-slate-800 rounded-xl flex items-center gap-4 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">
                            <span className="material-symbols-outlined">chat</span>
                        </div>
                        <div className="text-left">
                            <p className="font-bold text-sm dark:text-white">Live Chat</p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Available 9 AM - 5 PM</p>
                        </div>
                    </button>

                    <button
                        onClick={() => window.location.href = 'mailto:help@cardioai.com'}
                        className="w-full p-4 bg-slate-50 dark:bg-slate-800 rounded-xl flex items-center gap-4 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center text-orange-600 dark:text-orange-400">
                            <span className="material-symbols-outlined">mail</span>
                        </div>
                        <div className="text-left">
                            <p className="font-bold text-sm dark:text-white">Email Support</p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">help@cardioai.com</p>
                        </div>
                    </button>
                </div>
            </div>
        </div>
    );
};

const VideoConsultationModal = ({ appointment, onClose }: { appointment: Appointment, onClose: () => void }) => {
    const [isMuted, setIsMuted] = useState(false);
    const [isVideoOff, setIsVideoOff] = useState(false);
    const [transcript, setTranscript] = useState<string[]>([]);
    const [showScribe, setShowScribe] = useState(true);
    const [isSummarizing, setIsSummarizing] = useState(false);
    const [summary, setSummary] = useState<string | null>(null);
    const userVideoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // Web Speech API
    const recognitionRef = useRef<any>(null);

    useEffect(() => {
        // Start Local Video
        startCamera();

        // Simulate Doctor Joining Message
        setTimeout(() => {
            setTranscript(prev => [...prev, `Doctor: Hello! I'm ready to review your recent health assessment.`]);
        }, 2000);

        // Simulate Doctor Chatting periodically
        const interval = setInterval(() => {
            const phrases = [
                "Doctor: How have you been feeling since we last spoke?",
                "Doctor: I see your blood pressure logs are looking stable.",
                "Doctor: Any side effects from the new medication?",
                "Doctor: Remember to keep your sodium intake low."
            ];
            const randomPhrase = phrases[Math.floor(Math.random() * phrases.length)];
            setTranscript(prev => [...prev, randomPhrase]);
        }, 8000);

        // Setup Speech Recognition for User
        if ('webkitSpeechRecognition' in window) {
            const recognition = new (window as any).webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = (event: any) => {
                const text = event.results[event.results.length - 1][0].transcript;
                setTranscript(prev => [...prev, `You: ${text}`]);
            };

            recognition.start();
            recognitionRef.current = recognition;
        }

        return () => {
            clearInterval(interval);
            if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
            if (recognitionRef.current) recognitionRef.current.stop();
        };
    }, []);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            streamRef.current = stream;
            if (userVideoRef.current) {
                userVideoRef.current.srcObject = stream;
            }
        } catch (e) {
            console.error("Camera error:", e);
        }
    };

    const toggleMute = () => {
        if (streamRef.current) {
            streamRef.current.getAudioTracks().forEach(t => t.enabled = !isMuted);
            setIsMuted(!isMuted);
        }
    };

    const toggleVideo = () => {
        if (streamRef.current) {
            streamRef.current.getVideoTracks().forEach(t => t.enabled = !isVideoOff);
            setIsVideoOff(!isVideoOff);
        }
    };

    const endCall = async () => {
        // Stop media
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
        if (recognitionRef.current) recognitionRef.current.stop();

        setIsSummarizing(true);
        try {
            // Use backend API for consultation summary
            const result = await apiClient.generateInsight({
                user_name: 'Patient',
                vitals: {}
            });

            setSummary(result.insight || "Summary generated.");
        } catch (e) {
            console.error("Summarization failed", e);
            setSummary("Could not generate summary. Consultation ended.");
        } finally {
            setIsSummarizing(false);
        }
    };

    if (summary) {
        return (
            <div className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-4">
                <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-md shadow-2xl animate-in zoom-in-95">
                    <div className="flex items-center gap-3 mb-4 text-green-600">
                        <span className="material-symbols-outlined text-3xl">assignment_turned_in</span>
                        <h2 className="text-xl font-bold dark:text-white">Consultation Complete</h2>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-800/50 p-4 rounded-xl max-h-[60vh] overflow-y-auto mb-4 border border-slate-100 dark:border-slate-700">
                        <h3 className="text-xs font-bold text-slate-500 uppercase mb-2">AI Clinical Summary</h3>
                        <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{summary}</p>
                    </div>
                    <button onClick={onClose} className="w-full py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30">
                        Save to Records & Close
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[60] bg-slate-900 flex flex-col">
            {/* Main Video Area (Doctor) */}
            <div className="flex-1 relative overflow-hidden">
                {/* Loop a stock video of a doctor */}
                <video
                    src="https://videos.pexels.com/video-files/5452203/5452203-hd_1920_1080_25fps.mp4"
                    autoPlay
                    loop
                    muted
                    className="w-full h-full object-cover"
                />

                <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-lg flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-white text-sm font-bold">Dr. {appointment.doctorName}</span>
                </div>

                {/* Self View (PiP) */}
                <div className="absolute top-4 right-4 w-32 aspect-[3/4] bg-black rounded-xl overflow-hidden shadow-lg border-2 border-white/20">
                    {!isVideoOff ? (
                        <video ref={userVideoRef} autoPlay muted playsInline className="w-full h-full object-cover transform -scale-x-100" />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center bg-slate-800">
                            <span className="material-symbols-outlined text-slate-500">videocam_off</span>
                        </div>
                    )}
                </div>

                {/* AI Scribe Overlay */}
                {showScribe && (
                    <div className="absolute bottom-24 left-4 w-64 max-h-48 bg-black/60 backdrop-blur-md rounded-xl p-3 overflow-y-auto flex flex-col-reverse fade-mask">
                        {transcript.slice(-4).reverse().map((line, i) => (
                            <p key={i} className="text-xs text-white/90 mb-1">
                                <span className={`font-bold ${line.startsWith('You') ? 'text-blue-300' : 'text-green-300'}`}>
                                    {line.split(':')[0]}:
                                </span>
                                {line.split(':')[1]}
                            </p>
                        ))}
                        <div className="flex items-center gap-1 text-[10px] text-blue-300 uppercase font-bold tracking-wider mb-2 sticky top-0">
                            <span className="material-symbols-outlined text-sm animate-pulse">mic</span> AI Scribe Active
                        </div>
                    </div>
                )}
            </div>

            {/* Controls Bar */}
            <div className="h-20 bg-slate-900 flex items-center justify-center gap-6 px-4 shrink-0">
                <button
                    onClick={toggleMute}
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${isMuted ? 'bg-red-500 text-white' : 'bg-slate-700 text-white hover:bg-slate-600'}`}
                >
                    <span className="material-symbols-outlined">{isMuted ? 'mic_off' : 'mic'}</span>
                </button>
                <button
                    onClick={endCall}
                    className="w-16 h-16 rounded-full bg-red-600 flex items-center justify-center text-white shadow-lg hover:bg-red-500 transition-transform hover:scale-105"
                >
                    <span className="material-symbols-outlined text-3xl">call_end</span>
                </button>
                <button
                    onClick={toggleVideo}
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${isVideoOff ? 'bg-red-500 text-white' : 'bg-slate-700 text-white hover:bg-slate-600'}`}
                >
                    <span className="material-symbols-outlined">{isVideoOff ? 'videocam_off' : 'videocam'}</span>
                </button>
                <button
                    onClick={() => setShowScribe(!showScribe)}
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${showScribe ? 'bg-blue-500 text-white' : 'bg-slate-700 text-white hover:bg-slate-600'}`}
                    title="Toggle Transcription"
                >
                    <span className="material-symbols-outlined">subtitles</span>
                </button>
            </div>

            {/* Summarizing Loader */}
            {isSummarizing && (
                <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-[70]">
                    <span className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></span>
                    <p className="text-white font-bold">Generating Clinical Summary...</p>
                </div>
            )}
        </div>
    );
};

const AppointmentScreen: React.FC = () => {
  const navigate = useNavigate();
  const { showToast } = useToast();
  const confirmDialog = useConfirm();

  // --- State ---
  const [view, setView] = useState<'list' | 'detail'>('list');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSpecialty, setSelectedSpecialty] = useState('All');
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null);

  // Booking State
  const [appointmentType, setAppointmentType] = useState<'in-person' | 'video'>('in-person');
  const [currentMonthDate, setCurrentMonthDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedTime, setSelectedTime] = useState<string>('');

  // Intake & Triage State
  const [showIntakeModal, setShowIntakeModal] = useState(false);
  const [intakeData, setIntakeData] = useState<{reason: string, summary: string} | null>(null);

  // New Features State
  const [insuranceDetails, setInsuranceDetails] = useState({ provider: '', memberId: '', groupId: '' });
  const [isScanningInsurance, setIsScanningInsurance] = useState(false);
  const [shareChart, setShareChart] = useState(false);
  const insuranceInputRef = useRef<HTMLInputElement>(null);

  // Modal State
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [showReceptionist, setShowReceptionist] = useState(false);
  const [activeVideoCall, setActiveVideoCall] = useState<Appointment | null>(null);

  // Data from backend
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [latestBooking, setLatestBooking] = useState<Appointment | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [specialties, setSpecialties] = useState<string[]>(['All']);
  const [availableSlots, setAvailableSlots] = useState<string[]>([]);
  const [isLoadingProviders, setIsLoadingProviders] = useState(false);
  const [isLoadingSlots, setIsLoadingSlots] = useState(false);

  // --- Load providers from backend ---
  const loadProviders = useCallback(async () => {
      setIsLoadingProviders(true);
      try {
          const data = await apiClient.getProviders({
              specialty: selectedSpecialty !== 'All' ? selectedSpecialty : undefined,
              search: searchQuery || undefined,
          });
          setProviders(data);
      } catch (e) {
          console.error('Failed to load providers', e);
          showToast('Failed to load providers', 'error');
      } finally {
          setIsLoadingProviders(false);
      }
  }, [selectedSpecialty, searchQuery]);

  // --- Load specialties from backend ---
  useEffect(() => {
      (async () => {
          try {
              const result = await apiClient.getSpecialties();
              setSpecialties(result.specialties || ['All']);
          } catch (e) {
              console.error('Failed to load specialties', e);
          }
      })();
  }, []);

  // --- Load providers when filters change ---
  useEffect(() => {
      loadProviders();
  }, [loadProviders]);

  // --- Load user appointments from backend ---
  const loadAppointments = useCallback(async () => {
      try {
          const data = await apiClient.getUserAppointments(USER_ID);
          const mapped: Appointment[] = data.map((a: any) => ({
              id: a.appointment_id,
              doctorName: a.doctorName || a.doctor_name,
              specialty: a.specialty,
              date: a.date,
              time: a.time,
              type: a.type || a.appointment_type || 'in-person',
              location: a.location || '',
              rating: a.doctor_rating,
              summary: a.summary || a.consultation_summary || a.intake_summary || '',
          }));
          mapped.sort((a, b) => new Date(`${a.date}T${a.time}`).getTime() - new Date(`${b.date}T${b.time}`).getTime());
          setAppointments(mapped);
          // Also sync to localStorage for dashboard widget
          localStorage.setItem('user_appointments', JSON.stringify(mapped));
      } catch (e) {
          console.error('Failed to load appointments from API, trying localStorage', e);
          const saved = localStorage.getItem('user_appointments');
          if (saved) {
              try {
                  const parsed = JSON.parse(saved);
                  parsed.sort((a: Appointment, b: Appointment) => new Date(`${a.date}T${a.time}`).getTime() - new Date(`${b.date}T${b.time}`).getTime());
                  setAppointments(parsed);
              } catch (err) {
                  console.error("Failed to parse appointments", err);
              }
          }
      }
  }, []);

  useEffect(() => {
      loadAppointments();
  }, [loadAppointments]);

  // --- Load availability when provider + date change ---
  useEffect(() => {
      if (!selectedProvider || !selectedDate) {
          setAvailableSlots([]);
          return;
      }
      (async () => {
          setIsLoadingSlots(true);
          try {
              const result = await apiClient.getProviderAvailability(selectedProvider.id, selectedDate);
              setAvailableSlots(result.slots || []);
          } catch (e) {
              console.error('Failed to load availability', e);
              // Fallback: generate default weekday slots
              const day = new Date(selectedDate).getDay();
              if (day !== 0 && day !== 6) {
                  setAvailableSlots(['09:00', '10:00', '11:00', '14:00', '15:30']);
              } else {
                  setAvailableSlots([]);
              }
          } finally {
              setIsLoadingSlots(false);
          }
      })();
  }, [selectedProvider, selectedDate]);

  // --- Filtering is now handled by the backend API ---
  // providers state is already filtered from loadProviders()

  // --- Calendar Helpers ---
  const getDaysInMonth = (date: Date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const days = new Date(year, month + 1, 0).getDate();
    return Array.from({ length: days }, (_, i) => i + 1);
  };

  const getFirstDayOfMonth = (date: Date) => {
    return new Date(date.getFullYear(), date.getMonth(), 1).getDay();
  };

  const handleMonthChange = (increment: number) => {
    const newDate = new Date(currentMonthDate);
    newDate.setMonth(newDate.getMonth() + increment);
    setCurrentMonthDate(newDate);
    setSelectedDate('');
    setSelectedTime('');
  };

  const formatDate = (date: Date) => {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
  };

  const handleDateSelect = (day: number) => {
    const newDate = new Date(currentMonthDate.getFullYear(), currentMonthDate.getMonth(), day);
    setSelectedDate(formatDate(newDate));
    setSelectedTime('');
  };

  // --- Insurance Scanner ---
  const handleInsuranceScan = async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setIsScanningInsurance(true);
      const reader = new FileReader();
      reader.onloadend = async () => {
          const base64Data = (reader.result as string).split(',')[1];

          try {
              // Simplified insurance scan - would integrate with backend API in production
              setInsuranceDetails({
                  provider: 'Insurance Provider',
                  memberId: 'MEM-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
                  groupId: 'GRP-' + Math.random().toString(36).substr(2, 9).toUpperCase()
              });
          } catch (err) {
              console.error("Insurance Scan Error", err);
              showToast("Could not scan card. Please enter details manually.", 'error');
          } finally {
              setIsScanningInsurance(false);
          }
      };
      reader.readAsDataURL(file);
  };

  // --- Intake Handler ---
  const handleIntakeComplete = (reason: string, urgency: string, summary: string) => {
      setIntakeData({ reason, summary });
      setShowIntakeModal(false);
      handleConfirmBooking(summary);
  };

  const initiateBooking = () => {
      if (!intakeData) {
          setShowIntakeModal(true);
      } else {
          handleConfirmBooking(intakeData.summary);
      }
  };

  const handleConfirmBooking = async (medicalSummary?: string) => {
    if (!selectedProvider || !selectedDate || !selectedTime) return;

    let finalSummary = medicalSummary || '';
    let sharedChart: Record<string, any> | undefined;

    // Append shared chart data if enabled
    if (shareChart) {
        const savedAssessment = localStorage.getItem('last_assessment');
        if (savedAssessment) {
            const data = JSON.parse(savedAssessment);
            sharedChart = {
                bp: `${data.vitals.systolic}/80`,
                cholesterol: data.vitals.cholesterol,
                risk_level: data.risk,
                date: data.date,
            };
            finalSummary += `\n\n[SHARED CHART DATA]\nBP: ${data.vitals.systolic}/80\nCholesterol: ${data.vitals.cholesterol}\nRisk Level: ${data.risk}`;
        }
    }

    try {
        // Book via backend API
        const result = await apiClient.createAppointment(USER_ID, {
            provider_id: selectedProvider.id,
            date: selectedDate,
            time: selectedTime,
            appointment_type: appointmentType,
            reason: intakeData?.reason,
            intake_summary: finalSummary,
            shared_chart_data: sharedChart,
            insurance_provider: insuranceDetails.provider || undefined,
            insurance_member_id: insuranceDetails.memberId || undefined,
            insurance_group_id: insuranceDetails.groupId || undefined,
            estimated_cost: 150.0,
        });

        const newAppt: Appointment = {
            id: result.appointment_id,
            doctorName: result.doctorName || result.doctor_name || selectedProvider.name,
            specialty: selectedProvider.specialty,
            date: selectedDate,
            time: selectedTime,
            type: appointmentType,
            location: selectedProvider.clinicName,
            rating: selectedProvider.rating,
            summary: finalSummary,
        };

        setLatestBooking(newAppt);

        // Reload appointments from backend
        await loadAppointments();

        // Save notification
        const newNotification = {
            id: Date.now(),
            title: 'Appointment Confirmed',
            message: `Booked with Dr. ${selectedProvider.name} for ${selectedDate} at ${selectedTime}.`,
            time: 'Just now',
            icon: 'event_available',
            color: 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400',
            path: '/appointment'
        };
        const existingNotifs = JSON.parse(localStorage.getItem('user_notifications') || '[]');
        localStorage.setItem('user_notifications', JSON.stringify([newNotification, ...existingNotifs]));

        setShowSuccessModal(true);
        showToast('Appointment booked successfully!', 'success');
    } catch (error: any) {
        console.error('Booking failed', error);
        showToast(error?.message || 'Failed to book appointment. Please try again.', 'error');
    }
  };

  const resetFlow = () => {
      setShowSuccessModal(false);
      setSelectedProvider(null);
      setSelectedDate('');
      setSelectedTime('');
      setIntakeData(null);
      setInsuranceDetails({ provider: '', memberId: '', groupId: '' });
      setShareChart(false);
      setView('list');
      navigate('/dashboard');
  };

  // --- Render Views ---

  const renderSuccessModal = () => {
      if (!showSuccessModal || !latestBooking) return null;

      const addToGoogleCalendar = async () => {
          const { date, time, doctorName, location } = latestBooking;
          const startTime = `${date.replace(/-/g, '')}T${time.replace(':', '')}00`;
          let h = parseInt(time.split(':')[0]);
          const m = time.split(':')[1];
          const endTime = `${date.replace(/-/g, '')}T${(h+1) < 10 ? '0'+(h+1) : (h+1)}${m}00`;

          const title = encodeURIComponent(`Appointment with Dr. ${doctorName}`);
          const details = encodeURIComponent(`Cardiology appointment.\nSpecialty: ${latestBooking.specialty}\nType: ${latestBooking.type}`);
          const loc = encodeURIComponent(location);

          const url = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${title}&dates=${startTime}/${endTime}&details=${details}&location=${loc}`;
          // Use Capacitor Browser plugin on Android, fallback for web
          try {
            const { Browser } = await import('@capacitor/browser');
            await Browser.open({ url });
          } catch {
            window.open(url, '_system') || window.open(url, '_blank') || (window.location.href = url);
          }
      };

      return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-sm shadow-2xl transform transition-all scale-100">
                <div className="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="material-symbols-outlined text-green-500 text-3xl">check_circle</span>
                </div>
                <h2 className="text-xl font-bold text-center mb-2 dark:text-white">Booking Confirmed!</h2>
                <p className="text-slate-500 dark:text-slate-400 text-center mb-6 text-sm">Your appointment has been successfully scheduled.</p>

                <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-4 mb-4 space-y-3 border border-slate-100 dark:border-slate-700">
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 dark:text-slate-400 text-sm">Specialist</span>
                        <span className="font-semibold dark:text-white text-sm">Dr. {latestBooking.doctorName}</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 dark:text-slate-400 text-sm">Date & Time</span>
                        <span className="font-semibold dark:text-white text-sm">{latestBooking.date}, {latestBooking.time}</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 dark:text-slate-400 text-sm">Type</span>
                        <div className="flex items-center gap-1 font-semibold dark:text-white text-sm">
                            <span className="material-symbols-outlined text-xs">
                                {latestBooking.type === 'video' ? 'videocam' : 'location_on'}
                            </span>
                            <span className="capitalize">{latestBooking.type}</span>
                        </div>
                    </div>
                </div>

                <div className="space-y-3">
                    <button
                        onClick={addToGoogleCalendar}
                        className="w-full py-3 bg-white border border-slate-200 dark:bg-slate-800 dark:border-slate-700 text-slate-700 dark:text-white rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                    >
                        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Google_Calendar_icon_%282020%29.svg" alt="GCal" className="w-5 h-5" />
                        Add to Google Calendar
                    </button>

                    <div className="flex gap-3">
                        <button
                            onClick={resetFlow}
                            className="flex-1 py-3 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-white rounded-xl font-bold transition-colors"
                        >
                            Home
                        </button>
                        <button
                            onClick={() => { setShowSuccessModal(false); navigate('/appointment'); }}
                            className="flex-1 py-3 bg-primary hover:bg-primary-dark text-white rounded-xl font-bold transition-colors shadow-lg shadow-primary/20"
                        >
                            View All
                        </button>
                    </div>
                </div>
            </div>
        </div>
      );
  };

  const renderDetailView = () => {
    if (!selectedProvider) return null;

    const days = getDaysInMonth(currentMonthDate);
    const startDay = getFirstDayOfMonth(currentMonthDate);
    const todayStr = formatDate(new Date());

    // Data for Chart Preview
    const savedAssessment = localStorage.getItem('last_assessment');
    const healthData = savedAssessment ? JSON.parse(savedAssessment) : null;

    return (
        <div className="flex flex-col min-h-screen bg-background-light dark:bg-background-dark pb-32 animate-in slide-in-from-right duration-300 overflow-x-hidden">
            <div className="flex items-center p-4 bg-white dark:bg-card-dark sticky top-0 z-10 border-b border-slate-100 dark:border-slate-800 shadow-sm">
                <button onClick={() => { setView('list'); setIntakeData(null); }} className="p-2 -ml-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-900 dark:text-white transition-colors">
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>
                <h2 className="flex-1 text-center font-bold text-lg dark:text-white">Book Appointment</h2>
                <div className="w-10"></div>
            </div>

            <div className="p-4 space-y-6">
                {/* Doctor Profile Card */}
                <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-800 flex flex-col gap-4">
                    <div className="flex gap-4">
                        <img src={selectedProvider.photoUrl} alt={selectedProvider.name} className="w-20 h-20 rounded-full object-cover border-2 border-slate-100 dark:border-slate-700" />
                        <div className="flex-1">
                            <h3 className="text-xl font-bold dark:text-white">Dr. {selectedProvider.name}</h3>
                            <p className="text-primary font-medium text-sm">{selectedProvider.specialty}</p>
                            <p className="text-slate-500 text-xs mt-1">{selectedProvider.qualifications} • {selectedProvider.experienceYears}+ years exp</p>
                            <div className="flex items-center gap-1 mt-2 text-yellow-500 text-sm font-bold">
                                <span className="material-symbols-outlined filled text-sm">star</span>
                                {selectedProvider.rating}
                                <span className="text-slate-400 font-normal text-xs">({selectedProvider.reviewCount} reviews)</span>
                            </div>
                        </div>
                    </div>

                    <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed border-t border-slate-100 dark:border-slate-800 pt-3">
                        {selectedProvider.bio}
                    </p>

                    <div className="flex flex-wrap gap-2 pt-1">
                         {selectedProvider.telehealthAvailable && (
                             <span className="px-2 py-1 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 text-xs rounded-md font-medium flex items-center gap-1">
                                 <span className="material-symbols-outlined text-xs">videocam</span> Video Consult
                             </span>
                         )}
                         <span className="px-2 py-1 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs rounded-md font-medium flex items-center gap-1">
                             <span className="material-symbols-outlined text-xs">location_on</span> {selectedProvider.clinicName}
                         </span>
                    </div>
                </div>

                {/* Appointment Type */}
                <div>
                    <h3 className="font-bold text-lg mb-3 dark:text-white">Appointment Type</h3>
                    <div className="flex bg-slate-200 dark:bg-slate-800 p-1 rounded-xl">
                        <button
                            onClick={() => setAppointmentType('in-person')}
                            className={`flex-1 py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all ${
                                appointmentType === 'in-person'
                                ? 'bg-white dark:bg-card-dark text-slate-900 dark:text-white shadow-sm'
                                : 'text-slate-500 dark:text-slate-400'
                            }`}
                        >
                            <span className="material-symbols-outlined text-sm">local_hospital</span> In-person
                        </button>
                        <button
                             onClick={() => setAppointmentType('video')}
                             className={`flex-1 py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all ${
                                appointmentType === 'video'
                                ? 'bg-white dark:bg-card-dark text-slate-900 dark:text-white shadow-sm'
                                : 'text-slate-500 dark:text-slate-400'
                            }`}
                             disabled={!selectedProvider.telehealthAvailable}
                        >
                            <span className="material-symbols-outlined text-sm">videocam</span> Video
                        </button>
                    </div>
                </div>

                {/* Date Selection */}
                <div>
                    <h3 className="font-bold text-lg mb-3 dark:text-white">Select Date</h3>
                    <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-800">
                        <div className="flex justify-between items-center mb-4">
                            <button onClick={() => handleMonthChange(-1)} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full text-slate-600 dark:text-slate-300">
                                <span className="material-symbols-outlined">chevron_left</span>
                            </button>
                            <span className="font-bold dark:text-white">
                                {currentMonthDate.toLocaleString('default', { month: 'long', year: 'numeric' })}
                            </span>
                            <button onClick={() => handleMonthChange(1)} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full text-slate-600 dark:text-slate-300">
                                <span className="material-symbols-outlined">chevron_right</span>
                            </button>
                        </div>

                        <div className="grid grid-cols-7 text-center text-xs text-slate-400 mb-2 font-medium">
                            {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((d, i) => <div key={i}>{d}</div>)}
                        </div>

                        <div className="grid grid-cols-7 gap-1">
                            {Array.from({ length: startDay }).map((_, i) => <div key={`empty-${i}`}></div>)}
                            {days.map(day => {
                                const d = new Date(currentMonthDate.getFullYear(), currentMonthDate.getMonth(), day);
                                const dStr = formatDate(d);
                                const isSelected = dStr === selectedDate;
                                const isPast = dStr < todayStr;
                                const isToday = dStr === todayStr;

                                return (
                                    <button
                                        key={day}
                                        onClick={() => !isPast && handleDateSelect(day)}
                                        disabled={isPast}
                                        className={`
                                            h-9 w-9 rounded-full flex items-center justify-center text-sm transition-all mx-auto
                                            ${isSelected ? 'bg-primary text-white font-bold shadow-md' : ''}
                                            ${!isSelected && !isPast ? 'text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700' : ''}
                                            ${isPast ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed' : ''}
                                            ${isToday && !isSelected ? 'border border-primary text-primary font-bold' : ''}
                                        `}
                                    >
                                        {day}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {selectedDate && (
                    <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 space-y-6">
                        {/* Time Selection */}
                        <div>
                            <h3 className="font-bold text-lg mb-3 dark:text-white">Select Time</h3>
                            {isLoadingSlots ? (
                                <div className="flex items-center justify-center py-6 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
                                    <span className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin mr-2"></span>
                                    <span className="text-slate-500 text-sm">Loading available slots...</span>
                                </div>
                            ) : availableSlots.length > 0 ? (
                                <div className="grid grid-cols-3 sm:grid-cols-4 gap-3">
                                    {availableSlots.map(time => (
                                        <button
                                            key={time}
                                            onClick={() => setSelectedTime(time)}
                                            className={`py-2 px-1 rounded-xl text-sm font-medium border transition-all ${
                                                selectedTime === time
                                                ? 'bg-primary border-primary text-white shadow-md'
                                                : 'bg-white dark:bg-card-dark border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-primary/50'
                                            }`}
                                        >
                                            {time}
                                        </button>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-center py-6 bg-slate-50 dark:bg-slate-800/50 rounded-xl border-dashed border-2 border-slate-200 dark:border-slate-700">
                                    <p className="text-slate-500 text-sm">No available slots for this date.</p>
                                </div>
                            )}
                        </div>

                        {/* Payment & Insurance Section */}
                        {selectedTime && (
                            <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-800">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="font-bold text-lg dark:text-white flex items-center gap-2">
                                        <span className="material-symbols-outlined text-blue-500">credit_card</span>
                                        Payment & Insurance
                                    </h3>
                                    <button
                                        onClick={() => insuranceInputRef.current?.click()}
                                        disabled={isScanningInsurance}
                                        className="text-xs font-bold text-primary flex items-center gap-1 bg-primary/10 px-3 py-1.5 rounded-lg hover:bg-primary/20 transition-colors"
                                    >
                                        {isScanningInsurance ? (
                                            <span className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>
                                        ) : (
                                            <span className="material-symbols-outlined text-sm">photo_camera</span>
                                        )}
                                        Scan Card
                                    </button>
                                    <input
                                        type="file"
                                        ref={insuranceInputRef}
                                        accept="image/*"
                                        className="hidden"
                                        onChange={handleInsuranceScan}
                                    />
                                </div>
                                <div className="space-y-3">
                                    <input
                                        type="text"
                                        placeholder="Provider (e.g. Aetna)"
                                        value={insuranceDetails.provider}
                                        onChange={(e) => setInsuranceDetails({...insuranceDetails, provider: e.target.value})}
                                        className="w-full p-3 bg-slate-50 dark:bg-slate-800 rounded-xl border-none outline-none focus:ring-2 focus:ring-primary dark:text-white text-sm"
                                    />
                                    <div className="grid grid-cols-2 gap-3">
                                        <input
                                            type="text"
                                            placeholder="Member ID"
                                            value={insuranceDetails.memberId}
                                            onChange={(e) => setInsuranceDetails({...insuranceDetails, memberId: e.target.value})}
                                            className="w-full p-3 bg-slate-50 dark:bg-slate-800 rounded-xl border-none outline-none focus:ring-2 focus:ring-primary dark:text-white text-sm"
                                        />
                                        <input
                                            type="text"
                                            placeholder="Group ID"
                                            value={insuranceDetails.groupId}
                                            onChange={(e) => setInsuranceDetails({...insuranceDetails, groupId: e.target.value})}
                                            className="w-full p-3 bg-slate-50 dark:bg-slate-800 rounded-xl border-none outline-none focus:ring-2 focus:ring-primary dark:text-white text-sm"
                                        />
                                    </div>
                                    {insuranceDetails.provider && (
                                        <p className="text-xs text-green-500 flex items-center gap-1 mt-1">
                                            <span className="material-symbols-outlined text-xs">check_circle</span>
                                            Details verified
                                        </p>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Share Data Section */}
                        {selectedTime && (
                            <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <span className="material-symbols-outlined text-purple-500">folder_shared</span>
                                        <span className="font-bold text-sm dark:text-white">Share My Chart</span>
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" checked={shareChart} onChange={(e) => setShareChart(e.target.checked)} className="sr-only peer"/>
                                        <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-slate-600 peer-checked:bg-primary"></div>
                                    </label>
                                </div>
                                <p className="text-xs text-slate-500 mb-3">Allow Dr. {selectedProvider.name} to view your recent vitals and assessment history.</p>

                                {shareChart && healthData && (
                                    <div className="bg-slate-50 dark:bg-slate-800/50 p-3 rounded-xl border border-slate-200 dark:border-slate-700 animate-in fade-in">
                                        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Preview of Shared Data</p>
                                        <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-700 dark:text-slate-300">
                                            <div>BP: <span className="font-bold">{healthData.vitals.systolic}/80</span></div>
                                            <div>Chol: <span className="font-bold">{healthData.vitals.cholesterol}</span></div>
                                            <div>Risk: <span className={`font-bold ${healthData.risk === 'High Risk' ? 'text-red-500' : 'text-green-500'}`}>{healthData.risk}</span></div>
                                            <div>Date: {new Date(healthData.date).toLocaleDateString()}</div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="mt-auto p-4 bg-white dark:bg-card-dark border-t border-slate-100 dark:border-slate-800 sticky bottom-0 z-20">
                <div className="flex justify-between items-center mb-2">
                    <div className="text-xs text-slate-500">Total Estimated Cost</div>
                    <div className="text-lg font-bold dark:text-white">$150.00</div>
                </div>
                <button
                    onClick={initiateBooking}
                    disabled={!selectedDate || !selectedTime}
                    className="w-full py-4 bg-primary disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary-dark text-white rounded-xl font-bold shadow-lg shadow-primary/30 transition-all flex items-center justify-center gap-2"
                >
                    <span>{intakeData ? 'Confirm Booking' : 'Proceed to Intake'}</span>
                    {selectedDate && selectedTime && <span className="material-symbols-outlined text-sm">arrow_forward</span>}
                </button>
            </div>
        </div>
    );
  };

  const renderListView = () => (
    <div className="min-h-screen bg-background-light dark:bg-background-dark pb-24 overflow-x-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-background-light dark:bg-background-dark sticky top-0 z-10 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center">
            <button onClick={() => navigate('/dashboard')} className="p-2 -ml-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-900 dark:text-white transition-colors">
            <span className="material-symbols-outlined">arrow_back</span>
            </button>
            <h2 className="text-lg font-bold dark:text-white ml-2">Find a Specialist</h2>
        </div>
        <button
            onClick={() => setShowReceptionist(true)}
            className="flex items-center gap-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 px-3 py-1.5 rounded-full text-xs font-bold hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
        >
            <span className="material-symbols-outlined text-sm">support_agent</span>
            <span className="hidden sm:inline">Reception</span>
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Upcoming Appointments List */}
        {appointments.length > 0 && (
            <div className="mb-4">
                <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-2">My Bookings</h3>
                <div className="flex gap-3 overflow-x-auto no-scrollbar pb-2 snap-x">
                    {appointments.map((appt, i) => (
                        <div key={i} className="min-w-[280px] snap-center bg-gradient-to-br from-indigo-600 to-blue-600 dark:from-indigo-900 dark:to-blue-900 rounded-2xl p-4 text-white shadow-lg relative overflow-hidden flex-shrink-0">
                            <div className="flex justify-between items-start mb-3">
                                <div>
                                    <p className="text-indigo-100 text-xs font-bold uppercase tracking-wider mb-1">
                                        {new Date(`${appt.date}T${appt.time}`).getTime() > Date.now() ? 'Upcoming' : 'Past'}
                                    </p>
                                    <h3 className="font-bold text-base text-white">Dr. {appt.doctorName}</h3>
                                    <p className="text-xs text-indigo-100">{appt.specialty}</p>
                                </div>
                                <div className="w-8 h-8 bg-white/20 backdrop-blur-md rounded-lg flex items-center justify-center">
                                    <span className="material-symbols-outlined text-white text-sm">event</span>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <div className="bg-black/20 backdrop-blur-sm rounded-lg px-2 py-1 flex items-center gap-1">
                                    <span className="material-symbols-outlined text-[10px]">calendar_month</span>
                                    <span className="font-bold text-xs">{new Date(appt.date).toLocaleDateString(undefined, {month: 'short', day: 'numeric'})}</span>
                                </div>
                                <div className="bg-black/20 backdrop-blur-sm rounded-lg px-2 py-1 flex items-center gap-1">
                                    <span className="material-symbols-outlined text-[10px]">schedule</span>
                                    <span className="font-bold text-xs">{appt.time}</span>
                                </div>

                                {appt.type === 'video' && new Date(`${appt.date}T${appt.time}`).getTime() > Date.now() && (
                                    <button
                                        onClick={() => setActiveVideoCall(appt)}
                                        className="ml-auto bg-green-500 hover:bg-green-600 text-white rounded-lg px-3 py-1 flex items-center gap-1 transition-colors shadow-sm"
                                    >
                                        <span className="material-symbols-outlined text-[12px]">videocam</span>
                                        <span className="font-bold text-xs">Join</span>
                                    </button>
                                )}
                                {new Date(`${appt.date}T${appt.time}`).getTime() > Date.now() && (
                                    <button
                                        onClick={async (e) => {
                                            e.stopPropagation();
                                            const confirmed = await confirmDialog({
                                                title: 'Cancel Appointment',
                                                message: `Are you sure you want to cancel your appointment with ${appt.doctorName || 'the doctor'} on ${appt.date} at ${appt.time}? This action cannot be undone.`,
                                                confirmText: 'Yes, Cancel',
                                                cancelText: 'Keep It',
                                                variant: 'danger',
                                            });
                                            if (!confirmed) return;
                                            try {
                                                await apiClient.cancelAppointment(USER_ID, appt.id, 'Cancelled by patient');
                                                showToast('Appointment cancelled', 'success');
                                                await loadAppointments();
                                            } catch (err) {
                                                console.error('Cancel failed', err);
                                                showToast('Failed to cancel', 'error');
                                            }
                                        }}
                                        className="ml-auto bg-red-500/80 hover:bg-red-600 text-white rounded-lg px-3 py-1 flex items-center gap-1 transition-colors shadow-sm"
                                    >
                                        <span className="material-symbols-outlined text-[12px]">close</span>
                                        <span className="font-bold text-xs">Cancel</span>
                                    </button>
                                )}
                            </div>
                            {appt.summary && (
                                <div className="mt-3 pt-3 border-t border-white/10">
                                    <p className="text-[10px] text-indigo-100 line-clamp-2 italic">"{appt.summary.substring(0, 60)}..."</p>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        )}

        {/* Search */}
        <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-400">search</span>
            <input
                type="text"
                placeholder="Search by name or specialty"
                className="w-full pl-12 pr-4 h-12 rounded-xl bg-white dark:bg-slate-800 border-none focus:ring-2 focus:ring-primary outline-none dark:text-white shadow-sm"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
            />
        </div>

        {/* Filters */}
        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1">
            {specialties.map(spec => (
                <button
                    key={spec}
                    onClick={() => setSelectedSpecialty(spec)}
                    className={`px-4 h-10 rounded-full font-medium text-sm whitespace-nowrap transition-colors ${
                        selectedSpecialty === spec
                        ? 'bg-primary text-white shadow-md shadow-primary/20'
                        : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                    }`}
                >
                    {spec}
                </button>
            ))}
        </div>

        {/* Provider List */}
        <div className="space-y-4 pt-2">
            {isLoadingProviders ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                    <span className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin mb-3"></span>
                    <p className="text-slate-500 font-medium">Loading providers...</p>
                </div>
            ) : providers.length > 0 ? providers.map(provider => (
                <div
                    key={provider.id}
                    onClick={() => { setSelectedProvider(provider); setView('detail'); }}
                    className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 flex gap-4 cursor-pointer hover:border-primary/50 transition-colors group"
                >
                    <img src={provider.photoUrl} alt={provider.name} className="w-20 h-20 rounded-full object-cover bg-slate-200 border border-slate-100 dark:border-slate-700" />
                    <div className="flex-1 min-w-0">
                        <div className="flex justify-between items-start">
                            <div>
                                <h3 className="font-bold text-lg dark:text-white truncate">Dr. {provider.name}</h3>
                                <p className="text-primary text-sm font-medium">{provider.specialty}</p>
                            </div>
                            <div className="flex items-center gap-1 bg-yellow-50 dark:bg-yellow-900/20 px-1.5 py-0.5 rounded text-xs text-yellow-600 dark:text-yellow-400 font-bold">
                                <span className="material-symbols-outlined filled text-[10px]">star</span>
                                {provider.rating}
                            </div>
                        </div>

                        <div className="mt-2 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                             <span className="truncate flex-1">{provider.clinicName}</span>
                             {provider.telehealthAvailable && (
                                 <span className="shrink-0 material-symbols-outlined text-[14px] text-green-500" title="Telehealth Available">videocam</span>
                             )}
                        </div>

                        <div className="mt-3 pt-3 border-t border-slate-50 dark:border-slate-800 flex justify-between items-center">
                             <span className="text-xs text-slate-400 font-medium">Next Available: Today</span>
                             <span className="text-primary text-xs font-bold group-hover:underline">Book Now</span>
                        </div>
                    </div>
                </div>
            )) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                    <span className="material-symbols-outlined text-4xl text-slate-300 mb-2">person_search</span>
                    <p className="text-slate-500 font-medium">No specialists found</p>
                    <p className="text-slate-400 text-sm">Try adjusting your filters</p>
                </div>
            )}
        </div>
      </div>
    </div>
  );

  return (
    <>
        {view === 'list' && renderListView()}
        {view === 'detail' && renderDetailView()}
        {renderSuccessModal()}
        {showReceptionist && <ReceptionistModal onClose={() => setShowReceptionist(false)} />}
        {showIntakeModal && selectedProvider && (
            <IntakeModal
                doctorName={selectedProvider.name}
                onCancel={() => setShowIntakeModal(false)}
                onComplete={handleIntakeComplete}
            />
        )}
        {activeVideoCall && <VideoConsultationModal appointment={activeVideoCall} onClose={() => setActiveVideoCall(null)} />}
    </>
  );
};

export default AppointmentScreen;
