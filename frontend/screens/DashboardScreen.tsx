
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { HealthAssessment, Appointment, Device, Medication, FamilyMember } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import { apiClient, APIError } from '../services/apiClient';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const data = [
    { day: 'Mon', bpm: 68, note: 'Normal resting heart rate.' },
    { day: 'Tue', bpm: 72, note: 'Slightly elevated. Did you have coffee?' },
    { day: 'Wed', bpm: 70, note: 'Good stability.' },
    { day: 'Thu', bpm: 65, note: 'Excellent recovery rate observed.' },
    { day: 'Fri', bpm: 85, note: 'High activity detected (Cardio).' },
    { day: 'Sat', bpm: 75, note: 'Moderate active recovery.' },
    { day: 'Sun', bpm: 72, note: 'Baseline returned to normal.' },
    { day: 'Mon', bpm: 69, note: 'Start of new week.' },
    { day: 'Tue', bpm: 71, note: 'Consistent.' },
];

const DAD_VITALS = [
    { day: 'Mon', bpm: 82, note: 'High baseline.' },
    { day: 'Tue', bpm: 85, note: 'Medication missed morning.' },
    { day: 'Wed', bpm: 88, note: 'Stress reported.' },
    { day: 'Thu', bpm: 84, note: 'Slight improvement.' },
    { day: 'Fri', bpm: 90, note: 'Warning: Tachycardia alert.' },
    { day: 'Sat', bpm: 85, note: 'Stabilizing.' },
    { day: 'Sun', bpm: 86, note: 'High baseline persists.' },
];

interface InsightCache {
    date: string;
    text: string;
}

// --- Medical ID Modal Component ---
const MedicalIdModal = ({ onClose }: { onClose: () => void }) => {
    const [profile, setProfile] = useState<any>(null);

    useEffect(() => {
        const saved = localStorage.getItem('user_profile');
        if (saved) {
            setProfile(JSON.parse(saved));
        }
    }, []);

    const defaultProfile = {
        name: 'Eleanor Rigby',
        dob: '1980-01-01',
        bloodType: 'A+',
        conditions: ['Hypertension'],
        allergies: ['Penicillin'],
        medications: ['Lisinopril 10mg'],
        emergencyContact: {
            name: 'John Doe',
            relation: 'Spouse',
            phone: '+1 987 654 3210'
        },
        avatar: 'https://lh3.googleusercontent.com/aida-public/AB6AXuC8JJmFSNEDykVbLmg9GaDjI_y7oSrZg8hS9KI3YR7e3vQdQysk4FtU7xmAvLKhSuMQZgg2zbablylPhaXKCoy8vetGjpLe-Ty24fgpXbanV3G0gdxLOQp4UFEWDlaNETaNcWE1X-jhCKNT4bqUYPHtiTEZIBu24Ly5r-YP5vdBILXMcYIiLG6s8i1KztyEq0E4k79NTPODK1qXJhtVCURhe4x6JxRUzdlvshbonwupAWRLiXvZWsuODqHjdudOj9DAgtdsg0ScrbvE'
    };

    const data = profile ? { ...defaultProfile, ...profile, bloodType: profile.bloodType || 'A+' } : defaultProfile;

    return (
        <div className="fixed inset-0 z-[100] bg-slate-50 dark:bg-slate-900 flex flex-col animate-in slide-in-from-bottom duration-300">
            {/* Header */}
            <div className="bg-red-600 text-white p-4 pt-6 pb-6 shadow-lg rounded-b-3xl shrink-0">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h2 className="text-2xl font-bold flex items-center gap-2">
                            <span className="material-symbols-outlined filled text-3xl">medical_information</span>
                            Medical ID
                        </h2>
                        <p className="text-red-100 text-sm opacity-90">Emergency Information</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-10 h-10 bg-white/20 hover:bg-white/30 rounded-full flex items-center justify-center backdrop-blur-md transition-colors"
                    >
                        <span className="material-symbols-outlined text-white">close</span>
                    </button>
                </div>

                <div className="flex items-center gap-4 mt-2">
                    <img src={data.avatar} alt="Profile" className="w-20 h-20 rounded-full border-4 border-white/30 object-cover bg-slate-200" />
                    <div>
                        <h3 className="text-xl font-bold">{data.name}</h3>
                        <div className="flex gap-3 text-sm text-red-100 mt-1">
                            <span>DOB: {new Date(data.dob).toLocaleDateString()}</span>
                            <span className="w-px h-4 bg-white/40"></span>
                            <span className="font-bold bg-white text-red-600 px-2 rounded-md text-xs flex items-center">Blood: {data.bloodType}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-5">
                {/* Conditions */}
                <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border-l-4 border-red-500">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Medical Conditions</h4>
                    <div className="flex flex-wrap gap-2">
                        {data.conditions && data.conditions.length > 0 ? data.conditions.map((c: string, i: number) => (
                            <span key={i} className="px-3 py-1 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 rounded-lg text-sm font-bold border border-red-100 dark:border-red-900/30">
                                {c}
                            </span>
                        )) : <span className="text-slate-500 text-sm">None listed</span>}
                    </div>
                </div>

                {/* Allergies */}
                <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border-l-4 border-orange-500">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Allergies & Reactions</h4>
                    <div className="flex flex-wrap gap-2">
                        {data.allergies && data.allergies.length > 0 ? data.allergies.map((c: string, i: number) => (
                            <span key={i} className="px-3 py-1 bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 rounded-lg text-sm font-bold border border-orange-100 dark:border-orange-900/30 flex items-center gap-1">
                                <span className="material-symbols-outlined text-xs">warning</span> {c}
                            </span>
                        )) : <span className="text-slate-500 text-sm">None listed</span>}
                    </div>
                </div>

                {/* Medications */}
                <div className="bg-white dark:bg-card-dark p-4 rounded-2xl shadow-sm border-l-4 border-blue-500">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Current Medications</h4>
                    <ul className="space-y-2">
                        {data.medications && data.medications.length > 0 ? data.medications.map((m: any, i: number) => (
                            <li key={i} className="flex items-center gap-2 text-slate-700 dark:text-slate-200 text-sm">
                                <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                                {typeof m === 'string' ? m : `${m.name} ${m.dosage || ''}`}
                            </li>
                        )) : <li className="text-slate-500 text-sm">None listed</li>}
                    </ul>
                </div>
            </div>

            {/* Footer Action */}
            <div className="p-4 bg-white dark:bg-card-dark border-t border-slate-100 dark:border-slate-800 shrink-0">
                <p className="text-xs text-slate-400 font-bold uppercase mb-2">Emergency Contact</p>
                <a href={`tel:${data.emergencyContact.phone}`} className="w-full py-4 bg-red-600 hover:bg-red-700 active:scale-95 transition-all text-white rounded-xl font-bold flex items-center justify-between px-6 shadow-lg shadow-red-500/30">
                    <div className="flex flex-col items-start">
                        <span className="text-xs opacity-80 uppercase">{data.emergencyContact.relation}</span>
                        <span className="text-lg">{data.emergencyContact.name}</span>
                    </div>
                    <span className="material-symbols-outlined text-3xl">call</span>
                </a>
            </div>
        </div>
    );
};

const DashboardScreen: React.FC = () => {
    const { t } = useLanguage();
    const navigate = useNavigate();
    const [reminderSet, setReminderSet] = useState(false);
    const [assessment, setAssessment] = useState<HealthAssessment | null>(null);
    const [userName, setUserName] = useState('Alex');
    const [aiInsight, setAiInsight] = useState<string>(t('dashboard.loading_insight'));
    const [loadingInsight, setLoadingInsight] = useState(false);
    const [nextAppointment, setNextAppointment] = useState<Appointment | null>(null);

    // Real-time Data State
    const [liveHeartRate, setLiveHeartRate] = useState(72);
    const [steps, setSteps] = useState(5243);
    const [connectedDevice, setConnectedDevice] = useState<Device | null>(null);

    // Water Tracker State
    const [waterIntake, setWaterIntake] = useState(3);
    const WATER_GOAL = 8;

    // Medication State
    const [nextMed, setNextMed] = useState<Medication | null>(null);

    // Chart Interaction State
    const [selectedPoint, setSelectedPoint] = useState<any | null>(null);

    // Caretaker Mode
    const [caretakerMode, setCaretakerMode] = useState<string | null>(null);
    const [viewingProfile, setViewingProfile] = useState<FamilyMember | null>(null);

    // Notification & Modal State
    const [showNotifications, setShowNotifications] = useState(false);
    const [hasUnreadNotifications, setHasUnreadNotifications] = useState(true);
    const [showMedicalID, setShowMedicalID] = useState(false);
    const [insightExpanded, setInsightExpanded] = useState(false);

    const [notifications, setNotifications] = useState<any[]>([
        { id: 1, title: 'Hydration Alert', message: 'Time to drink water!', time: '10m ago', icon: 'water_drop', color: 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400', path: '/dashboard' },
    ]);

    useEffect(() => {
        // Update loading text when language changes
        setAiInsight(t('dashboard.loading_insight'));

        // Check for Caretaker Mode
        const viewingId = localStorage.getItem('active_profile_mode');
        setCaretakerMode(viewingId);

        if (viewingId) {
            if (viewingId === 'dad_01') {
                setViewingProfile({ id: 'dad_01', name: 'Robert Rigby', relation: 'Father', avatar: 'https://randomuser.me/api/portraits/men/85.jpg', accessLevel: 'read-only', status: 'Warning', lastActive: '10m ago' });
                setUserName('Robert');
                setAssessment({
                    date: new Date().toISOString(),
                    score: 45,
                    risk: 'High Risk',
                    details: 'Alert: Blood pressure consistently high. Missed medication doses recorded.',
                    vitals: { systolic: 150, cholesterol: 240 }
                });
                setSteps(1200); // Low steps
                setLiveHeartRate(88);
                // Mock missed med
                setNextMed({
                    id: 'm1', name: 'Lisinopril', dosage: '10mg', frequency: 'Daily', times: ['09:00'], takenToday: [false], instructions: 'Missed Dose'
                });
                setAiInsight("Robert has missed his morning medication. Consider calling him to check in.");
                setLoadingInsight(false);
                return; // Skip normal load
            }
        }

        // Normal User Load
        const saved = localStorage.getItem('last_assessment');
        let loadedAssessment: HealthAssessment | null = null;
        if (saved) {
            loadedAssessment = JSON.parse(saved);
            setAssessment(loadedAssessment);
        }

        const savedProfile = localStorage.getItem('user_profile');
        let profileName = 'Alex';
        if (savedProfile) {
            const profile = JSON.parse(savedProfile);
            profileName = profile.name.split(' ')[0];
            setUserName(profileName);
        }

        // 2. Check for Connected Devices
        const devicesRaw = localStorage.getItem('connected_devices');
        if (devicesRaw) {
            const devices = JSON.parse(devicesRaw) as Device[];
            const activeWatch = devices.find(d => d.status === 'connected' && d.type === 'watch');
            if (activeWatch) {
                setConnectedDevice(activeWatch);
            }
        }

        // 3. Load Medications
        const medsRaw = localStorage.getItem('user_medications');
        if (medsRaw) {
            const meds = JSON.parse(medsRaw) as Medication[];
            if (meds.length > 0) {
                const untaken = meds.find(m => m.takenToday.some(t => !t));
                setNextMed(untaken || meds[0]);
            }
        }

        // 4. Load Real Notifications & Appointments
        const savedNotifs = JSON.parse(localStorage.getItem('user_notifications') || '[]');
        if (savedNotifs.length > 0) {
            setNotifications(prev => {
                const existingIds = new Set(prev.map(n => n.id));
                const newUnique = savedNotifs.filter((n: any) => !existingIds.has(n.id));
                return [...newUnique, ...prev];
            });
            setHasUnreadNotifications(true);
        }

        const savedApptsRaw = localStorage.getItem('user_appointments');
        if (savedApptsRaw) {
            const appts: Appointment[] = JSON.parse(savedApptsRaw);
            if (appts.length > 0) {
                appts.sort((a, b) => new Date(`${a.date}T${a.time}`).getTime() - new Date(`${b.date}T${b.time}`).getTime());
                const futureAppts = appts.filter(a => new Date(`${a.date}T${a.time}`).getTime() > new Date().getTime());

                if (futureAppts.length > 0) {
                    setNextAppointment(futureAppts[0]);
                    setReminderSet(true);
                } else if (appts.length > 0) {
                    setNextAppointment(appts[appts.length - 1]);
                }
            }
        }

        // 5. Handle AI Insight Generation
        generateDailyInsight(profileName, loadedAssessment);
    }, [t]);

    // Simulator for Live Data
    useEffect(() => {
        if (connectedDevice || caretakerMode) {
            const interval = setInterval(() => {
                setLiveHeartRate(prev => {
                    const base = caretakerMode ? 85 : 70;
                    const change = Math.floor(Math.random() * 5) - 2;
                    let val = prev + change;
                    if (val < base - 5) val = base - 5;
                    if (val > base + 20) val = base + 20;
                    return val;
                });
                if (!caretakerMode) {
                    setSteps(prev => prev + Math.floor(Math.random() * 3));
                }
            }, 2000);
            return () => clearInterval(interval);
        }
    }, [connectedDevice, caretakerMode]);

    const generateDailyInsight = async (name: string, data: HealthAssessment | null) => {
        const today = new Date().toDateString();
        const cacheKey = 'daily_insight_cache';
        const cachedRaw = localStorage.getItem(cacheKey);

        if (cachedRaw) {
            const cached: InsightCache = JSON.parse(cachedRaw);
            if (cached.date === today) {
                setAiInsight(cached.text);
                return;
            }
        }

        setLoadingInsight(true);
        try {
            // Prepare vitals data from assessment
            const vitals: any = {};
            if (data?.vitals?.systolic) {
                vitals.blood_pressure = `${data.vitals.systolic}/80`;
            }
            if (data?.vitals?.cholesterol) {
                vitals.cholesterol = data.vitals.cholesterol;
            }

            // Call backend API proxy instead of direct GoogleGenAI
            const response = await apiClient.generateInsight({
                user_name: name,
                vitals: Object.keys(vitals).length > 0 ? vitals : {
                    blood_pressure: "N/A"
                },
                activities: [],
                medications: []
            });

            const text = response.insight?.trim() || getDefaultInsight(data);
            setAiInsight(text);
            localStorage.setItem(cacheKey, JSON.stringify({ date: today, text }));

        } catch (error) {
            console.error("AI Insight Error:", error);
            if (error instanceof APIError) {
                console.error(`API Error ${error.status}: ${error.message}`);
            }
            setAiInsight(getDefaultInsight(data));
        } finally {
            setLoadingInsight(false);
        }
    };

    const getDefaultInsight = (data: HealthAssessment | null) => {
        if (!data) return "Complete your assessment to receive personalized AI health insights about your heart health trends.";
        if (data.risk === 'High Risk') return "Your recent assessment indicates potential high risk factors. It is recommended to consult a specialist soon.";
        if (data.risk === 'Moderate Risk') return "Focusing on a heart-healthy lifestyle and increasing daily steps can help improve your score.";
        return "Great job! Your latest assessment shows low risk. Maintenance is key—keep up your current healthy habits.";
    };

    const getAppointmentDateParts = (dateStr: string) => {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return { day: '21', month: 'DEC' };
        return {
            day: d.getDate(),
            month: d.toLocaleString('default', { month: 'short' })
        };
    };

    const handleChartClick = (data: any) => {
        if (data && data.activePayload && data.activePayload[0]) {
            setSelectedPoint(data.activePayload[0].payload);
        }
    };

    return (
        <div className="p-4 space-y-6 overflow-x-hidden pb-24">
            {/* Caretaker Banner */}
            {caretakerMode && viewingProfile && (
                <div className="bg-orange-500 text-white px-4 py-3 rounded-xl flex items-center justify-between shadow-lg animate-in slide-in-from-top duration-300">
                    <div className="flex items-center gap-3">
                        <span className="material-symbols-outlined filled">visibility</span>
                        <div>
                            <p className="text-xs font-bold uppercase tracking-wider opacity-80">Caretaker Mode</p>
                            <p className="font-bold">Viewing: {viewingProfile.name}</p>
                        </div>
                    </div>
                    <Link to="/profile" className="bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors">
                        Exit
                    </Link>
                </div>
            )}

            {/* Header */}
            <div className="flex justify-between items-center py-2 relative z-30">
                <div className="flex items-center gap-3">
                    <Link to="/profile" className="w-10 h-10 rounded-full bg-cover bg-center border-2 border-primary cursor-pointer hover:opacity-80 transition-opacity block" style={{ backgroundImage: `url("${viewingProfile ? viewingProfile.avatar : 'https://picsum.photos/100/100'}")` }}></Link>
                    <div>
                        <h1 className="text-xl font-bold dark:text-white">
                            {caretakerMode ? viewingProfile?.name : `${t('dashboard.greeting')}, ${userName}`}
                        </h1>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                            {caretakerMode ? 'Read-Only View' : t('dashboard.welcome_back')}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {/* Medical ID Button */}
                    <button
                        onClick={() => setShowMedicalID(true)}
                        className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 flex items-center justify-center hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors shadow-sm"
                        title="Emergency Medical ID"
                    >
                        <span className="material-symbols-outlined filled">medical_information</span>
                    </button>

                    {/* Notifications */}
                    <div className="relative">
                        <button
                            onClick={() => { setShowNotifications(!showNotifications); if (showNotifications) setHasUnreadNotifications(false); }}
                            className="p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 relative hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                        >
                            <span className="material-symbols-outlined">notifications</span>
                            {hasUnreadNotifications && <span className="absolute top-2 right-2 w-2.5 h-2.5 bg-red-500 border-2 border-white dark:border-slate-800 rounded-full"></span>}
                        </button>

                        {showNotifications && (
                            <>
                                <div className="fixed inset-0 z-40" onClick={() => { setShowNotifications(false); setHasUnreadNotifications(false); }}></div>
                                <div className="absolute right-0 top-12 w-[calc(100vw-2rem)] max-w-80 bg-white dark:bg-card-dark rounded-2xl shadow-xl border border-slate-100 dark:border-slate-800 z-50 animate-in fade-in zoom-in-95 duration-200 origin-top-right overflow-hidden">
                                    <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
                                        <h3 className="font-bold text-sm dark:text-white">{t('dashboard.notifications')}</h3>
                                        <button onClick={() => { setShowNotifications(false); setHasUnreadNotifications(false); }} className="text-slate-400 hover:text-slate-600">
                                            <span className="material-symbols-outlined text-sm">close</span>
                                        </button>
                                    </div>
                                    <div className="max-h-[300px] overflow-y-auto">
                                        {notifications.length > 0 ? notifications.map(n => (
                                            <div
                                                key={n.id}
                                                onClick={() => {
                                                    setShowNotifications(false);
                                                    setHasUnreadNotifications(false);
                                                    if (n.path) navigate(n.path);
                                                }}
                                                className="p-4 border-b border-slate-50 dark:border-slate-800/50 flex gap-3 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
                                            >
                                                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${n.color}`}>
                                                    <span className="material-symbols-outlined text-lg">{n.icon}</span>
                                                </div>
                                                <div>
                                                    <div className="flex justify-between items-start gap-2">
                                                        <p className="text-sm font-bold text-slate-800 dark:text-white">{n.title}</p>
                                                        <p className="text-[10px] text-slate-400 whitespace-nowrap">{n.time}</p>
                                                    </div>
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{n.message}</p>
                                                </div>
                                            </div>
                                        )) : (
                                            <div className="p-6 text-center text-slate-500 text-xs">No notifications</div>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => { setShowNotifications(false); setHasUnreadNotifications(false); }}
                                        className="w-full p-3 text-center text-xs font-bold text-primary hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors border-t border-slate-100 dark:border-slate-800"
                                    >
                                        Mark all as read
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* AI Health Insight */}
            <div className={`bg-gradient-to-br ${caretakerMode ? 'from-orange-600 to-red-600' : 'from-indigo-500 to-purple-600'} rounded-2xl p-4 shadow-lg text-white relative overflow-hidden`}>
                <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-10 -mt-10 blur-2xl"></div>
                <div className="flex items-start gap-3 relative z-10">
                    <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center shrink-0">
                        {loadingInsight ? (
                            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        ) : (
                            <span className="material-symbols-outlined text-white">auto_awesome</span>
                        )}
                    </div>
                    <div className="flex-1 min-w-0">
                        <h3 className="font-bold text-lg mb-1">{caretakerMode ? 'Caretaker Alert' : t('dashboard.insight_title')}</h3>
                        <div className={`relative ${!insightExpanded ? 'max-h-[4.5rem] overflow-hidden' : ''}`}>
                            <div className="text-sm text-indigo-100 leading-relaxed ai-insight-content prose prose-sm prose-invert max-w-none
                                [&_p]:mb-1.5 [&_p]:last:mb-0
                                [&_ul]:mb-1.5 [&_ul]:pl-4 [&_ul]:list-disc
                                [&_ol]:mb-1.5 [&_ol]:pl-4 [&_ol]:list-decimal
                                [&_li]:mb-0.5 [&_li]:text-indigo-100
                                [&_strong]:text-white [&_strong]:font-semibold
                                [&_h1]:text-base [&_h1]:font-bold [&_h1]:mb-1 [&_h1]:text-white
                                [&_h2]:text-sm [&_h2]:font-bold [&_h2]:mb-1 [&_h2]:text-white
                                [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mb-1 [&_h3]:text-white
                                [&_blockquote]:border-l-2 [&_blockquote]:border-white/30 [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-indigo-200
                                [&_code]:bg-white/10 [&_code]:px-1 [&_code]:rounded [&_code]:text-xs
                                [&_hr]:border-white/20 [&_hr]:my-2
                            ">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {aiInsight}
                                </ReactMarkdown>
                            </div>
                            {!insightExpanded && aiInsight.length > 150 && (
                                <div className={`absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t ${caretakerMode ? 'from-orange-600/90' : 'from-indigo-600/90'} to-transparent pointer-events-none`}></div>
                            )}
                        </div>
                        {aiInsight.length > 150 && (
                            <button
                                onClick={() => setInsightExpanded(!insightExpanded)}
                                className="mt-1.5 text-xs font-bold text-white/80 hover:text-white flex items-center gap-1 transition-colors"
                            >
                                {insightExpanded ? 'Show less' : 'Read more'}
                                <span className="material-symbols-outlined text-xs">{insightExpanded ? 'expand_less' : 'expand_more'}</span>
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Heart Health Assessment Summary */}
            {assessment ? (
                <div className={`bg-card-light dark:bg-card-dark rounded-2xl p-4 shadow-sm border ${caretakerMode ? 'border-red-200 dark:border-red-900/50' : 'border-slate-100 dark:border-slate-800'}`}>
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h2 className="text-lg font-bold dark:text-white">{t('dashboard.risk_score')}</h2>
                            <div className="flex items-center gap-2 mt-1">
                                <span className={`text-sm font-bold ${assessment.risk === 'Low Risk' ? 'text-green-500' :
                                    assessment.risk === 'Moderate Risk' ? 'text-yellow-500' : 'text-red-500'
                                    }`}>
                                    {assessment.risk}
                                </span>
                                <span className="text-slate-300">•</span>
                                <span className="text-xs text-slate-500 dark:text-slate-400">Latest</span>
                            </div>
                        </div>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${assessment.score >= 80 ? 'bg-green-500/20 text-green-500' :
                            assessment.score >= 50 ? 'bg-yellow-500/20 text-yellow-500' : 'bg-red-500/20 text-red-500'
                            }`}>
                            <span className="font-bold text-sm">{assessment.score}</span>
                        </div>
                    </div>
                    <p className="text-sm text-slate-600 dark:text-slate-300 mb-4 line-clamp-2">
                        {assessment.details}
                    </p>
                    <Link to="/assessment" className={`block w-full py-3 ${caretakerMode ? 'bg-slate-200 dark:bg-slate-700 text-slate-500 pointer-events-none' : 'bg-primary hover:bg-primary-dark text-white'} rounded-xl text-center font-semibold transition-colors`}>
                        {caretakerMode ? t('dashboard.assessment_locked') : t('dashboard.view_report')}
                    </Link>
                </div>
            ) : (
                <div className="bg-card-light dark:bg-card-dark rounded-2xl p-6 shadow-sm border border-slate-100 dark:border-slate-800 text-center">
                    <div className="w-14 h-14 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center mx-auto mb-3 text-primary">
                        <span className="material-symbols-outlined text-2xl">health_metrics</span>
                    </div>
                    <h2 className="text-lg font-bold dark:text-white mb-2">{t('dashboard.check_health')}</h2>
                    <p className="text-sm text-slate-500 mb-4">{t('dashboard.check_health_desc')}</p>
                    <Link to="/assessment" className="block w-full py-3 bg-primary hover:bg-primary-dark text-white rounded-xl font-bold transition-colors">
                        {t('dashboard.start_assessment')}
                    </Link>
                </div>
            )}

            {/* Vitals Grid with Interactive Chart */}
            <div className="grid grid-cols-2 gap-4">
                {/* Heart Rate Card */}
                <div className="col-span-2 bg-white dark:bg-card-dark rounded-3xl p-6 shadow-lg border border-slate-100 dark:border-slate-800 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full -mr-16 -mt-16 blur-3xl pointer-events-none"></div>

                    <div className="flex justify-between items-center mb-6 relative z-10">
                        <div>
                            <h3 className="text-sm font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                {t('dashboard.heart_rate')}
                                {(connectedDevice || caretakerMode) && (
                                    <span className={`flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full ${caretakerMode ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'} animate-pulse`}>
                                        <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                                        LIVE
                                    </span>
                                )}
                            </h3>
                            <div className="flex items-baseline gap-3 mt-1">
                                <span className={`text-4xl font-black tracking-tight ${caretakerMode && liveHeartRate > 100 ? 'text-red-500' : 'text-slate-900 dark:text-white'}`}>
                                    {liveHeartRate > 0 ? liveHeartRate : '72'}
                                </span>
                                <span className="text-sm font-bold text-slate-400">BPM</span>
                                {connectedDevice && (
                                    <span className="text-xs text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-md flex items-center gap-1">
                                        <span className="material-symbols-outlined text-[10px]">watch</span>
                                        {connectedDevice.name}
                                    </span>
                                )}
                            </div>
                        </div>
                        <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${caretakerMode ? 'bg-red-50 text-red-500' : 'bg-red-50 dark:bg-red-900/20 text-red-500'} ${(connectedDevice || caretakerMode) ? 'animate-pulse' : ''}`}>
                            <span className="material-symbols-outlined text-2xl">favorite</span>
                        </div>
                    </div>

                    <div className="h-[180px] w-full min-w-0 min-h-0 relative z-10">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0} debounce={50}>
                            <AreaChart data={caretakerMode ? DAD_VITALS : data} onClick={handleChartClick}>
                                <defs>
                                    <linearGradient id="colorBpm" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={caretakerMode ? "#ef4444" : "#ec4899"} stopOpacity={0.2} />
                                        <stop offset="95%" stopColor={caretakerMode ? "#ef4444" : "#ec4899"} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontSize: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                                    itemStyle={{ color: '#fff', fontWeight: 'bold' }}
                                    cursor={{ stroke: caretakerMode ? "#ef4444" : "#ec4899", strokeWidth: 1, strokeDasharray: '4 4' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="bpm"
                                    stroke={caretakerMode ? "#ef4444" : "#ec4899"}
                                    strokeWidth={4}
                                    fillOpacity={1}
                                    fill="url(#colorBpm)"
                                    activeDot={{ r: 6, strokeWidth: 0 }}
                                />
                                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontWeight: 500 }} dy={10} interval={0} />
                            </AreaChart>
                        </ResponsiveContainer>
                        <p className="text-[10px] text-slate-400 text-center mt-2 font-medium opacity-0 group-hover:opacity-100 transition-opacity">Tap points for details</p>
                    </div>

                    {/* Context Modal Overlay for Chart Click */}
                    {selectedPoint && (
                        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-10 flex items-center justify-center p-4 rounded-2xl animate-in fade-in duration-200" onClick={() => setSelectedPoint(null)}>
                            <div className="bg-white dark:bg-slate-900 p-4 rounded-xl w-full max-w-xs shadow-xl" onClick={e => e.stopPropagation()}>
                                <div className="flex justify-between items-start mb-2">
                                    <h4 className="font-bold text-slate-900 dark:text-white">{selectedPoint.day} Heart Rate</h4>
                                    <button onClick={() => setSelectedPoint(null)}><span className="material-symbols-outlined text-slate-400">close</span></button>
                                </div>
                                <p className="text-2xl font-bold text-primary mb-2">{selectedPoint.bpm} <span className="text-sm font-normal text-slate-500">BPM</span></p>
                                <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded-lg text-xs text-slate-600 dark:text-slate-300">
                                    <strong>Context:</strong> {selectedPoint.note}
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Small Stats Cards */}
                {connectedDevice || caretakerMode ? (
                    <div className="bg-white dark:bg-card-dark rounded-3xl p-5 shadow-lg border border-slate-100 dark:border-slate-800 flex flex-col justify-between hover:scale-[1.02] transition-transform">
                        <div className="flex justify-between items-start">
                            <div className="w-10 h-10 rounded-xl bg-orange-50 dark:bg-orange-900/20 text-orange-500 flex items-center justify-center">
                                <span className={`material-symbols-outlined text-xl`}>directions_walk</span>
                            </div>
                            <span className={`text-xs font-bold px-2 py-1 rounded-full ${steps >= 8000 ? 'bg-green-100 text-green-600' : 'bg-slate-100 text-slate-500'}`}>
                                {Math.round((steps / 8000) * 100)}%
                            </span>
                        </div>
                        <div className="mt-4">
                            <p className="text-2xl font-black text-slate-900 dark:text-white">{steps.toLocaleString()}</p>
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t('dashboard.steps')}</p>
                        </div>
                    </div>
                ) : (
                    <div className="bg-white dark:bg-card-dark rounded-3xl p-5 shadow-lg border border-slate-100 dark:border-slate-800 hover:scale-[1.02] transition-transform">
                        <div className="flex justify-between items-start mb-4">
                            <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 text-indigo-500 flex items-center justify-center">
                                <span className="material-symbols-outlined text-xl">monitor_heart</span>
                            </div>
                            {assessment && (
                                <span className={`text-xs font-bold px-2 py-1 rounded-full ${assessment.vitals.systolic < 120 ? 'bg-green-100 text-green-600' : 'bg-yellow-100 text-yellow-600'}`}>
                                    Normal
                                </span>
                            )}
                        </div>
                        <p className="text-2xl font-black text-slate-900 dark:text-white">{assessment ? `${assessment.vitals.systolic}/80` : '118/75'}</p>
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t('dashboard.bp')} <span className="text-[10px] normal-case opacity-70">(mmHg)</span></p>
                    </div>
                )}

                {/* Medication Card (if active) */}
                {nextMed ? (
                    <Link to="/medications" className={`rounded-2xl p-4 shadow-sm border flex flex-col justify-between cursor-pointer hover:shadow-md transition-all ${caretakerMode && nextMed.instructions?.includes('Missed')
                        ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                        : 'bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-100 dark:border-blue-800/30'
                        }`}>
                        <div className="flex justify-between items-start">
                            <p className={`text-xs font-bold uppercase tracking-wider mb-1 ${caretakerMode && nextMed.instructions?.includes('Missed') ? 'text-red-600 dark:text-red-400' : 'text-blue-700 dark:text-blue-300'}`}>
                                {caretakerMode ? t('dashboard.missed_dose') : t('dashboard.next_dose')}
                            </p>
                            <span className={`material-symbols-outlined text-base ${caretakerMode && nextMed.instructions?.includes('Missed') ? 'text-red-500' : 'text-blue-500'}`}>
                                {caretakerMode ? 'warning' : 'medication'}
                            </span>
                        </div>
                        <div>
                            <p className="text-lg font-bold text-slate-900 dark:text-white truncate">{nextMed.name}</p>
                            <p className={`text-xs mt-0.5 ${caretakerMode ? 'text-red-500 dark:text-red-300' : 'text-slate-500 dark:text-slate-400'}`}>
                                {nextMed.times[0]} {caretakerMode && '- MISSED'}
                            </p>
                        </div>
                    </Link>
                ) : (
                    <div className="bg-card-light dark:bg-card-dark rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-800">
                        <p className="text-xs text-slate-500 mb-1">{t('dashboard.cholesterol')}</p>
                        <p className="text-xl font-bold dark:text-white">{assessment ? assessment.vitals.cholesterol : '175'}</p>
                        <p className="text-xs text-yellow-500 flex items-center gap-1">
                            <span className="material-symbols-outlined text-[10px]">warning</span> {t('dashboard.slightly_high')}
                        </p>
                    </div>
                )}
            </div>

            {/* Water Tracker */}
            <div className="bg-card-light dark:bg-card-dark rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-800">
                <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-500">
                            <span className="material-symbols-outlined text-sm">water_drop</span>
                        </div>
                        <div>
                            <h3 className="text-sm font-bold dark:text-white">{t('dashboard.hydration')}</h3>
                            <p className="text-xs text-slate-500">{waterIntake} / {WATER_GOAL} {t('dashboard.glasses')}</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setWaterIntake(prev => Math.min(prev + 1, WATER_GOAL))}
                        className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center hover:bg-blue-600 transition-colors shadow-lg shadow-blue-500/30"
                    >
                        <span className="material-symbols-outlined text-lg">add</span>
                    </button>
                </div>
                <div className="flex justify-between gap-1">
                    {Array.from({ length: WATER_GOAL }).map((_, i) => (
                        <div
                            key={i}
                            onClick={() => setWaterIntake(i + 1)}
                            className={`h-8 flex-1 rounded-md transition-all duration-300 cursor-pointer ${i < waterIntake
                                ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)] scale-y-100'
                                : 'bg-slate-100 dark:bg-slate-800 scale-y-75 hover:bg-slate-200 dark:hover:bg-slate-700'
                                }`}
                        ></div>
                    ))}
                </div>
            </div>

            {/* Quick Actions */}
            {!caretakerMode && (
                <div>
                    <h3 className="font-bold text-lg mb-4 dark:text-white flex items-center gap-2">
                        <span className="material-symbols-outlined text-primary">bolt</span>
                        {t('dashboard.quick_actions')}
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        <Link to="/assessment" className="bg-white dark:bg-card-dark p-5 rounded-2xl flex flex-col items-center gap-3 hover:shadow-lg transition-all border border-slate-100 dark:border-slate-800 group relative overflow-hidden">
                            <div className="absolute inset-0 bg-red-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="w-14 h-14 rounded-2xl bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                                <span className="material-symbols-outlined text-2xl">health_metrics</span>
                            </div>
                            <div className="text-center relative z-10">
                                <p className="font-bold text-sm text-slate-800 dark:text-white mb-0.5">{t('dashboard.heart_check')}</p>
                                <p className="text-[10px] text-slate-500 dark:text-slate-400">{t('dashboard.heart_check_desc')}</p>
                            </div>
                        </Link>
                        <Link to="/chat" className="bg-white dark:bg-card-dark p-5 rounded-2xl flex flex-col items-center gap-3 hover:shadow-lg transition-all border border-slate-100 dark:border-slate-800 group relative overflow-hidden">
                            <div className="absolute inset-0 bg-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="w-14 h-14 rounded-2xl bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                                <span className="material-symbols-outlined text-2xl">smart_toy</span>
                            </div>
                            <div className="text-center relative z-10">
                                <p className="font-bold text-sm text-slate-800 dark:text-white mb-0.5">{t('dashboard.ai_chat')}</p>
                                <p className="text-[10px] text-slate-500 dark:text-slate-400">{t('dashboard.ai_chat_desc')}</p>
                            </div>
                        </Link>
                        <Link to="/medications" className="bg-white dark:bg-card-dark p-5 rounded-2xl flex flex-col items-center gap-3 hover:shadow-lg transition-all border border-slate-100 dark:border-slate-800 group relative overflow-hidden">
                            <div className="absolute inset-0 bg-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="w-14 h-14 rounded-2xl bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                                <span className="material-symbols-outlined text-2xl">pill</span>
                            </div>
                            <div className="text-center relative z-10">
                                <p className="font-bold text-sm text-slate-800 dark:text-white mb-0.5">{t('dashboard.meds_manage')}</p>
                                <p className="text-[10px] text-slate-500 dark:text-slate-400">{t('dashboard.meds_desc')}</p>
                            </div>
                        </Link>
                        <Link to="/documents" className="bg-white dark:bg-card-dark p-5 rounded-2xl flex flex-col items-center gap-3 hover:shadow-lg transition-all border border-slate-100 dark:border-slate-800 group relative overflow-hidden">
                            <div className="absolute inset-0 bg-green-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="w-14 h-14 rounded-2xl bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                                <span className="material-symbols-outlined text-2xl">folder_open</span>
                            </div>
                            <div className="text-center relative z-10">
                                <p className="font-bold text-sm text-slate-800 dark:text-white mb-0.5">Documents</p>
                                <p className="text-[10px] text-slate-500 dark:text-slate-400">Medical records</p>
                            </div>
                        </Link>
                    </div>
                </div>
            )}

            {/* Upcoming Appointment Widget */}
            {nextAppointment && !caretakerMode && (
                <div className="bg-card-light dark:bg-card-dark rounded-2xl p-5 shadow-sm border border-slate-100 dark:border-slate-800 mb-20 animate-in slide-in-from-bottom-4 duration-500">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="font-bold text-lg dark:text-white">{t('dashboard.upcoming_appt')}</h3>
                        <button
                            onClick={() => setReminderSet(!reminderSet)}
                            className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-lg transition-colors ${reminderSet ? 'bg-primary/20 text-primary' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'}`}
                        >
                            <span className={`material-symbols-outlined text-sm ${reminderSet ? 'filled' : ''}`}>notifications_active</span>
                            {reminderSet ? t('dashboard.reminder_set') : t('dashboard.set_reminder')}
                        </button>
                    </div>
                    <div className="flex items-center gap-4">
                        {(() => {
                            const { day, month } = getAppointmentDateParts(nextAppointment.date);
                            return (
                                <div className="bg-slate-800 rounded-xl p-3 text-center min-w-[60px]">
                                    <span className="block text-xs text-blue-400 font-bold uppercase">{month}</span>
                                    <span className="block text-xl font-bold text-white">{day}</span>
                                </div>
                            );
                        })()}
                        <div className="flex-1">
                            <h4 className="font-bold text-slate-900 dark:text-white">Dr. {nextAppointment.doctorName}</h4>
                            <p className="text-xs text-slate-500 truncate">{nextAppointment.specialty}</p>
                            <p className="text-xs text-slate-400 mt-1 flex items-center gap-1">
                                <span className="material-symbols-outlined text-[10px]">{nextAppointment.type === 'video' ? 'videocam' : 'location_on'}</span>
                                {nextAppointment.time} • {nextAppointment.type}
                            </p>
                        </div>
                        <Link to="/appointment" className="w-10 h-10 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                            <span className="material-symbols-outlined">chevron_right</span>
                        </Link>
                    </div>
                </div>
            )}

            {!nextAppointment && (
                <div className="mb-20"></div>
            )}

            {/* Render Medical ID Modal */}
            {showMedicalID && <MedicalIdModal onClose={() => setShowMedicalID(false)} />}
        </div>
    );
};

export default DashboardScreen;
