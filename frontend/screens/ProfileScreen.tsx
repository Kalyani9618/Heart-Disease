
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { badgesData } from '../data/gamification';
import { FamilyMember } from '../types';
import { useToast } from '../components/Toast';
import { apiClient } from '../services/apiClient';
import { useAuth } from '../hooks/useAuth';

// --- Types ---
interface UserProfile {
  id: string;
  name: string;
  email: string;
  phone: string;
  dob: string;
  gender: string;
  conditions: string[];
  allergies: string[];
  medications: string[];
  emergencyContact: {
    name: string;
    relation: string;
    phone: string;
  };
  avatar: string;
}

const DEFAULT_PROFILE: UserProfile = {
  id: 'P-123456',
  name: 'Eleanor Rigby',
  email: 'e.rigby@email.com',
  phone: '+1 123 456 7890',
  dob: '1980-01-01',
  gender: 'Female',
  conditions: ['Hypertension'],
  allergies: ['Penicillin'],
  medications: [],
  emergencyContact: {
    name: 'John Doe',
    relation: 'Spouse',
    phone: '+1 987 654 3210'
  },
  avatar: 'https://lh3.googleusercontent.com/aida-public/AB6AXuC8JJmFSNEDykVbLmg9GaDjI_y7oSrZg8hS9KI3YR7e3vQdQysk4FtU7xmAvLKhSuMQZgg2zbablylPhaXKCoy8vetGjpLe-Ty24fgpXbanV3G0gdxLOQp4UFEWDlaNETaNcWE1X-jhCKNT4bqUYPHtiTEZIBu24Ly5r-YP5vdBILXMcYIiLG6s8i1KztyEq0E4k79NTPODK1qXJhtVCURhe4x6JxRUzdlvshbonwupAWRLiXvZWsuODqHjdudOj9DAgtdsg0ScrbvE'
};

const FAMILY_MEMBERS: FamilyMember[] = [
    { id: 'dad_01', name: 'Robert Rigby', relation: 'Father', avatar: 'https://randomuser.me/api/portraits/men/85.jpg', accessLevel: 'read-only', status: 'Warning', lastActive: '10m ago' },
    { id: 'mom_01', name: 'Martha Rigby', relation: 'Mother', avatar: 'https://randomuser.me/api/portraits/women/66.jpg', accessLevel: 'read-only', status: 'Stable', lastActive: '2h ago' }
];

// --- Modal Components ---

const PhotoEditModal = ({
  onSave,
  onClose
}: {
  onSave: (photoUrl: string) => void,
  onClose: () => void
}) => {
  const { showToast } = useToast();
  const [mode, setMode] = useState<'menu' | 'camera'>('menu');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      // Cleanup stream on unmount
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (typeof reader.result === 'string') {
          onSave(reader.result);
          onClose();
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleChooseFromGallery = async () => {
    try {
      const { Camera, CameraResultType, CameraSource } = await import('@capacitor/camera');
      const photo = await Camera.getPhoto({
        quality: 80,
        resultType: CameraResultType.DataUrl,
        source: CameraSource.Photos,
        width: 512,
        height: 512,
      });
      if (photo.dataUrl) {
        onSave(photo.dataUrl);
        onClose();
        return;
      }
    } catch (capError: any) {
      if (capError?.message?.includes('cancel') || capError?.message?.includes('Cancel')) return;
      // Fall back to file input
    }
    fileInputRef.current?.click();
  };

  const startCamera = async () => {
    try {
      // Try Capacitor Camera first (opens native camera app on Android)
      const { Camera, CameraResultType, CameraSource } = await import('@capacitor/camera');
      const photo = await Camera.getPhoto({
        quality: 80,
        resultType: CameraResultType.DataUrl,
        source: CameraSource.Camera,
        direction: 'front' as any,
        width: 512,
        height: 512,
      });
      if (photo.dataUrl) {
        onSave(photo.dataUrl);
        onClose();
        return;
      }
    } catch (capError: any) {
      // If user cancelled or Capacitor not available, fall back to WebView camera
      if (capError?.message?.includes('cancel') || capError?.message?.includes('Cancel')) return;
      console.debug('Capacitor camera not available, using WebView:', capError);
    }

    // Fallback: WebView getUserMedia
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      setStream(mediaStream);
      setMode('camera');
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.play();
        }
      }, 100);
    } catch (err) {
      console.error("Error accessing camera:", err);
      showToast("Could not access camera. Please check permissions.", 'error');
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Flip horizontally for mirror effect if using front camera (optional, usually expected)
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        onSave(dataUrl);

        // Stop stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        onClose();
      }
    }
  };

  const stopAndClose = () => {
      if (stream) {
          stream.getTracks().forEach(track => track.stop());
      }
      onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200" onClick={stopAndClose}>
      <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 w-full max-w-sm shadow-2xl overflow-hidden" onClick={e => e.stopPropagation()}>

        {mode === 'menu' && (
            <>
                <h3 className="text-xl font-bold dark:text-white mb-6 text-center">Change Profile Photo</h3>
                <div className="space-y-3">
                    <button
                        onClick={handleChooseFromGallery}
                        className="w-full p-4 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl flex items-center gap-4 hover:bg-slate-100 dark:hover:bg-slate-700/80 transition-colors"
                    >
                        <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">
                            <span className="material-symbols-outlined">photo_library</span>
                        </div>
                        <span className="font-bold text-slate-700 dark:text-white">Choose from Gallery</span>
                    </button>
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileChange}
                    />

                    <button
                        onClick={startCamera}
                        className="w-full p-4 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl flex items-center gap-4 hover:bg-slate-100 dark:hover:bg-slate-700/80 transition-colors"
                    >
                        <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">
                            <span className="material-symbols-outlined">photo_camera</span>
                        </div>
                        <span className="font-bold text-slate-700 dark:text-white">Take Photo</span>
                    </button>
                </div>
                <button onClick={onClose} className="w-full mt-6 py-3 text-slate-500 font-bold hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors">
                    Cancel
                </button>
            </>
        )}

        {mode === 'camera' && (
            <div className="flex flex-col items-center">
                <h3 className="text-lg font-bold dark:text-white mb-4">Take a Photo</h3>
                <div className="relative w-full aspect-square bg-black rounded-2xl overflow-hidden mb-6">
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform -scale-x-100"></video>
                    <canvas ref={canvasRef} className="hidden"></canvas>
                </div>
                <div className="flex gap-4 w-full">
                    <button onClick={() => { setMode('menu'); if(stream) stream.getTracks().forEach(t => t.stop()); }} className="flex-1 py-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl">
                        Back
                    </button>
                    <button onClick={capturePhoto} className="flex-1 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 flex items-center justify-center gap-2">
                        <span className="material-symbols-outlined">camera</span> Capture
                    </button>
                </div>
            </div>
        )}

      </div>
    </div>
  );
};

const EditPersonalModal = ({
  profile,
  onSave,
  onClose
}: {
  profile: UserProfile,
  onSave: (p: UserProfile) => void,
  onClose: () => void
}) => {
  const [formData, setFormData] = useState(profile);

  const handleChange = (field: keyof UserProfile, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
      <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 w-full max-w-sm shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold dark:text-white">Edit Personal Details</h3>
            <button onClick={onClose}><span className="material-symbols-outlined text-slate-500">close</span></button>
        </div>

        <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
            <div>
                <label className="text-xs font-bold text-slate-500 uppercase">Full Name</label>
                <input
                    type="text"
                    value={formData.name}
                    onChange={e => handleChange('name', e.target.value)}
                    className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
                />
            </div>
            <div>
                <label className="text-xs font-bold text-slate-500 uppercase">Email</label>
                <input
                    type="email"
                    value={formData.email}
                    onChange={e => handleChange('email', e.target.value)}
                    className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
                />
            </div>
            <div>
                <label className="text-xs font-bold text-slate-500 uppercase">Phone</label>
                <input
                    type="tel"
                    value={formData.phone}
                    onChange={e => handleChange('phone', e.target.value)}
                    className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
                />
            </div>
            <div className="grid grid-cols-2 gap-3">
                <div>
                    <label className="text-xs font-bold text-slate-500 uppercase">Date of Birth</label>
                    <input
                        type="date"
                        value={formData.dob}
                        onChange={e => handleChange('dob', e.target.value)}
                        className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
                    />
                </div>
                <div>
                    <label className="text-xs font-bold text-slate-500 uppercase">Gender</label>
                    <select
                        value={formData.gender}
                        onChange={e => handleChange('gender', e.target.value)}
                        className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary appearance-none"
                    >
                        <option>Male</option>
                        <option>Female</option>
                        <option>Other</option>
                    </select>
                </div>
            </div>
        </div>

        <button
            onClick={() => onSave(formData)}
            className="w-full mt-6 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 hover:bg-primary-dark transition-colors"
        >
            Save Changes
        </button>
      </div>
    </div>
  );
};

const AddItemModal = ({
  title,
  placeholder,
  onSave,
  onClose
}: {
  title: string,
  placeholder: string,
  onSave: (val: string) => void,
  onClose: () => void
}) => {
  const [val, setVal] = useState('');
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
      <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 w-full max-w-sm shadow-2xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-xl font-bold dark:text-white mb-4">{title}</h3>
        <input
            type="text"
            placeholder={placeholder}
            value={val}
            autoFocus
            onChange={e => setVal(e.target.value)}
            className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary mb-4"
        />
        <div className="flex gap-3">
            <button onClick={onClose} className="flex-1 py-3 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-bold rounded-xl">Cancel</button>
            <button onClick={() => { if(val) onSave(val); }} className="flex-1 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30">Add</button>
        </div>
      </div>
    </div>
  );
};

const EditContactModal = ({
  contact,
  onSave,
  onClose
}: {
  contact: UserProfile['emergencyContact'],
  onSave: (c: UserProfile['emergencyContact']) => void,
  onClose: () => void
}) => {
  const [data, setData] = useState(contact);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
      <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 w-full max-w-sm shadow-2xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-xl font-bold dark:text-white mb-4">Emergency Contact</h3>
        <div className="space-y-3">
            <input
                type="text"
                placeholder="Name"
                value={data.name}
                onChange={e => setData({...data, name: e.target.value})}
                className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
            />
            <input
                type="text"
                placeholder="Relationship"
                value={data.relation}
                onChange={e => setData({...data, relation: e.target.value})}
                className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
            />
            <input
                type="tel"
                placeholder="Phone Number"
                value={data.phone}
                onChange={e => setData({...data, phone: e.target.value})}
                className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
            />
        </div>
        <button
            onClick={() => onSave(data)}
            className="w-full mt-6 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30"
        >
            Save Contact
        </button>
      </div>
    </div>
  );
};

// --- Main Component ---

const ProfileScreen: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { showToast } = useToast();
  const [activeTab, setActiveTab] = useState<'Personal' | 'Medical' | 'Achievements' | 'Family' | 'Settings'>('Personal');
  const [profile, setProfile] = useState<UserProfile>(DEFAULT_PROFILE);
  const [activeCaretakerId, setActiveCaretakerId] = useState<string | null>(null);
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>(FAMILY_MEMBERS);
  const [loading, setLoading] = useState(true);

  // Modal States
  const [showEditPersonal, setShowEditPersonal] = useState(false);
  const [showAddCondition, setShowAddCondition] = useState(false);
  const [showAddAllergy, setShowAddAllergy] = useState(false);
  const [showEditContact, setShowEditContact] = useState(false);
  const [showPhotoModal, setShowPhotoModal] = useState(false);

  // Load from API with localStorage fallback
  useEffect(() => {
    const loadProfile = async () => {
      const userId = user?.id || localStorage.getItem('user_id') || 'default';
      try {
        const data = await apiClient.getProfile(userId);
        if (data && data.name) {
          setProfile({ ...DEFAULT_PROFILE, ...data });
          // Cache in localStorage
          localStorage.setItem('user_profile', JSON.stringify(data));
        } else {
          // Fall back to localStorage
          const saved = localStorage.getItem('user_profile');
          if (saved) setProfile({ ...DEFAULT_PROFILE, ...JSON.parse(saved) });
        }
      } catch (err) {
        console.warn('Failed to load profile from API, using localStorage', err);
        const saved = localStorage.getItem('user_profile');
        if (saved) setProfile({ ...DEFAULT_PROFILE, ...JSON.parse(saved) });
      }

      // Load family members from API
      try {
        const members = await apiClient.getFamilyMembers(userId);
        if (members && members.length > 0) {
          setFamilyMembers(members.map(m => ({
            id: m.id,
            name: m.name,
            relation: m.relation,
            avatar: m.avatar || '',
            accessLevel: (m.accessLevel as any) || 'read-only',
            status: (m.status as any) || 'Stable',
            lastActive: m.lastActive || ''
          })));
        }
      } catch {
        // Keep default family members
      }

      setLoading(false);
    };

    loadProfile();

    const caretakerMode = localStorage.getItem('active_profile_mode');
    if (caretakerMode) {
        setActiveCaretakerId(caretakerMode);
    }
  }, [user]);

  // Save to backend + localStorage helper
  const updateProfile = async (newProfile: UserProfile) => {
    setProfile(newProfile);
    localStorage.setItem('user_profile', JSON.stringify(newProfile));

    const userId = user?.id || localStorage.getItem('user_id') || 'default';
    try {
      await apiClient.updateProfile(userId, {
        name: newProfile.name,
        email: newProfile.email,
        phone: newProfile.phone,
        dob: newProfile.dob,
        gender: newProfile.gender,
      });

      // Update emergency contact
      await apiClient.updateEmergencyContact(userId, newProfile.emergencyContact);
    } catch (err) {
      console.warn('Failed to sync profile to backend', err);
    }
  };

  const handleTabChange = (tab: 'Personal' | 'Medical' | 'Achievements' | 'Family' | 'Settings') => {
    if (tab === 'Settings') {
      navigate('/settings');
    } else {
      setActiveTab(tab);
    }
  };

  const removeItem = (type: 'conditions' | 'allergies', index: number) => {
      const item = profile[type][index];
      const newList = [...profile[type]];
      newList.splice(index, 1);
      const newProfile = { ...profile, [type]: newList };
      setProfile(newProfile);
      localStorage.setItem('user_profile', JSON.stringify(newProfile));

      // Sync with backend
      const userId = user?.id || localStorage.getItem('user_id') || 'default';
      if (type === 'conditions') {
        apiClient.removeCondition(userId, item).catch(err => console.warn('Failed to sync remove condition', err));
      } else {
        apiClient.removeAllergy(userId, item).catch(err => console.warn('Failed to sync remove allergy', err));
      }
  };

  const addItem = (type: 'conditions' | 'allergies', value: string) => {
      const newProfile = { ...profile, [type]: [...profile[type], value] };
      setProfile(newProfile);
      localStorage.setItem('user_profile', JSON.stringify(newProfile));
      setShowAddCondition(false);
      setShowAddAllergy(false);

      // Sync with backend
      const userId = user?.id || localStorage.getItem('user_id') || 'default';
      if (type === 'conditions') {
        apiClient.addCondition(userId, value).catch(err => console.warn('Failed to sync add condition', err));
      } else {
        apiClient.addAllergy(userId, value).catch(err => console.warn('Failed to sync add allergy', err));
      }
  };

  const toggleCaretakerMode = (member: FamilyMember) => {
      if (activeCaretakerId === member.id) {
          localStorage.removeItem('active_profile_mode');
          setActiveCaretakerId(null);
      } else {
          localStorage.setItem('active_profile_mode', member.id);
          setActiveCaretakerId(member.id);
          navigate('/dashboard');
      }
  };

  return (
    <div className="relative flex h-auto min-h-screen w-full flex-col group/design-root overflow-x-hidden bg-background-light dark:bg-background-dark font-sans pb-24">
      {/* Top App Bar */}
      <div className="flex items-center bg-background-light dark:bg-background-dark p-4 pb-2 justify-between sticky top-0 z-10 border-b border-slate-200 dark:border-slate-800">
        <div className="flex size-10 shrink-0 items-center justify-center">
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center justify-center rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 w-10 h-10 transition-colors text-gray-800 dark:text-white"
          >
            <span className="material-symbols-outlined">arrow_back</span>
          </button>
        </div>
        <h2 className="text-gray-900 dark:text-white text-lg font-bold leading-tight tracking-[-0.015em] flex-1 text-center">
          My Profile
        </h2>
        <div className="flex size-10 shrink-0 items-center justify-center"></div>
      </div>

      {/* Profile Header */}
      <div className="flex p-4 @container">
        <div className="flex w-full flex-col gap-4 @[520px]:flex-row @[520px]:justify-between @[520px]:items-center">
          <div className="flex items-center gap-4">
            <div className="relative group cursor-pointer" onClick={() => setShowPhotoModal(true)}>
                <div
                  className="bg-center bg-no-repeat aspect-square bg-cover rounded-full h-24 w-24 border-4 border-white dark:border-slate-800 shadow-sm transition-opacity group-hover:opacity-90"
                  style={{backgroundImage: `url("${profile.avatar}")`}}
                ></div>
                <div className="absolute bottom-0 right-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center border-2 border-white dark:border-slate-900 shadow-sm">
                    <span className="material-symbols-outlined text-white text-sm">edit</span>
                </div>
            </div>

            <div className="flex flex-col justify-center">
              <p className="text-gray-900 dark:text-white text-[22px] font-bold leading-tight tracking-[-0.015em]">
                {profile.name}
              </p>
              <p className="text-gray-500 dark:text-gray-400 text-base font-normal leading-normal">
                ID: {profile.id}
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowEditPersonal(true)}
            className="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 bg-gray-200 dark:bg-slate-800 text-gray-800 dark:text-white text-sm font-bold leading-normal tracking-[0.015em] w-full max-w-[480px] @[480px]:w-auto hover:bg-gray-300 dark:hover:bg-slate-700 transition-colors gap-2"
          >
            <span className="material-symbols-outlined text-sm">edit</span>
            <span className="truncate">Edit Profile</span>
          </button>
        </div>
      </div>

      {/* Segmented Buttons */}
      <div className="flex px-4 py-3">
        <div className="flex h-10 flex-1 items-center justify-center rounded-lg bg-gray-200 dark:bg-slate-800 p-1 overflow-x-auto no-scrollbar">
          {['Personal', 'Medical', 'Achievements', 'Family', 'Settings'].map((tab) => (
            <label
              key={tab}
              onClick={() => handleTabChange(tab as any)}
              className={`flex cursor-pointer h-full grow items-center justify-center overflow-hidden rounded-lg px-2 transition-all duration-200 whitespace-nowrap ${
                activeTab === tab
                  ? 'bg-white dark:bg-slate-700 shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-primary dark:text-primary font-bold'
                  : 'text-gray-600 dark:text-gray-400 font-medium hover:bg-black/5 dark:hover:bg-white/5'
              } text-sm leading-normal`}
            >
              <span className="truncate px-1">{tab}</span>
              <input
                className="invisible w-0 absolute"
                name="profile-section"
                type="radio"
                value={tab}
                checked={activeTab === tab}
                readOnly
              />
            </label>
          ))}
        </div>
      </div>

      <div className="px-4 py-2">
        <div className="flex flex-col gap-3">
          {/* Personal Details Section */}
          {activeTab === 'Personal' && (
            <div className="flex flex-col rounded-xl border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 p-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-gray-600 dark:text-gray-400">person</span>
                  <h3 className="text-gray-900 dark:text-white text-base font-semibold">Personal Details</h3>
                </div>
              </div>
              <div className="flex flex-col gap-4">
                {[
                  { label: "Full Name", value: profile.name },
                  { label: "Date of Birth", value: new Date(profile.dob).toLocaleDateString() },
                  { label: "Gender", value: profile.gender },
                  { label: "Email", value: profile.email },
                  { label: "Phone", value: profile.phone }
                ].map((item, index) => (
                  <React.Fragment key={index}>
                    <div className="flex items-center justify-between">
                      <p className="text-gray-500 dark:text-gray-400 text-sm">{item.label}</p>
                      <p className="text-gray-800 dark:text-white text-sm font-medium">{item.value}</p>
                    </div>
                    {index < 4 && <div className="w-full h-px bg-gray-200 dark:bg-slate-800"></div>}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}

          {/* Accordions for Medical Info */}
          {activeTab === 'Medical' && (
            <div className="flex flex-col gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
              {/* Medical History */}
              <details className="flex flex-col rounded-xl border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 px-4 py-2 group" open>
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-2">
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-gray-600 dark:text-gray-400">medical_information</span>
                    <p className="text-gray-900 dark:text-white text-base font-semibold">Medical History</p>
                  </div>
                  <span className="material-symbols-outlined text-gray-600 dark:text-gray-400 group-open:rotate-180 transition-transform">expand_more</span>
                </summary>

                <div className="py-2 border-t border-gray-200 dark:border-slate-800 mt-2 space-y-4">
                    {/* Conditions */}
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <p className="text-gray-500 dark:text-gray-400 text-sm font-bold uppercase tracking-wider">Conditions</p>
                            <button onClick={() => setShowAddCondition(true)} className="text-primary text-xs font-bold flex items-center gap-1 hover:underline">
                                <span className="material-symbols-outlined text-xs">add</span> Add
                            </button>
                        </div>
                        {profile.conditions.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                                {profile.conditions.map((item, i) => (
                                    <span key={i} className="inline-flex items-center gap-1 px-3 py-1 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-full text-xs font-medium">
                                        {item}
                                        <button onClick={() => removeItem('conditions', i)} className="hover:text-red-800 dark:hover:text-red-200"><span className="material-symbols-outlined text-[10px]">close</span></button>
                                    </span>
                                ))}
                            </div>
                        ) : (
                            <p className="text-slate-400 text-xs italic">No conditions listed.</p>
                        )}
                    </div>

                    <div className="w-full h-px bg-gray-200 dark:bg-slate-800"></div>

                    {/* Allergies */}
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <p className="text-gray-500 dark:text-gray-400 text-sm font-bold uppercase tracking-wider">Allergies</p>
                            <button onClick={() => setShowAddAllergy(true)} className="text-primary text-xs font-bold flex items-center gap-1 hover:underline">
                                <span className="material-symbols-outlined text-xs">add</span> Add
                            </button>
                        </div>
                        {profile.allergies.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                                {profile.allergies.map((item, i) => (
                                    <span key={i} className="inline-flex items-center gap-1 px-3 py-1 bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 rounded-full text-xs font-medium">
                                        {item}
                                        <button onClick={() => removeItem('allergies', i)} className="hover:text-orange-800 dark:hover:text-orange-200"><span className="material-symbols-outlined text-[10px]">close</span></button>
                                    </span>
                                ))}
                            </div>
                        ) : (
                            <p className="text-slate-400 text-xs italic">No allergies listed.</p>
                        )}
                    </div>
                </div>
              </details>

              {/* Emergency Contact */}
              <details className="flex flex-col rounded-xl border border-gray-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 px-4 py-2 group" open>
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-2">
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-gray-600 dark:text-gray-400">contact_emergency</span>
                    <p className="text-gray-900 dark:text-white text-base font-semibold">Emergency Contact</p>
                  </div>
                  <span className="material-symbols-outlined text-gray-600 dark:text-gray-400 group-open:rotate-180 transition-transform">expand_more</span>
                </summary>
                <div className="flex flex-col gap-4 py-2 border-t border-gray-200 dark:border-slate-800 mt-2">
                  <div className="flex items-center justify-between">
                    <p className="text-gray-500 dark:text-gray-400 text-sm">Name</p>
                    <p className="text-gray-800 dark:text-white text-sm font-medium">{profile.emergencyContact.name}</p>
                  </div>
                  <div className="w-full h-px bg-gray-200 dark:bg-slate-800"></div>
                  <div className="flex items-center justify-between">
                    <p className="text-gray-500 dark:text-gray-400 text-sm">Relationship</p>
                    <p className="text-gray-800 dark:text-white text-sm font-medium">{profile.emergencyContact.relation}</p>
                  </div>
                  <div className="w-full h-px bg-gray-200 dark:bg-slate-800"></div>
                  <div className="flex items-center justify-between">
                    <p className="text-gray-500 dark:text-gray-400 text-sm">Phone</p>
                    <div className="flex items-center gap-2">
                        <p className="text-gray-800 dark:text-white text-sm font-medium">{profile.emergencyContact.phone}</p>
                        <a href={`tel:${profile.emergencyContact.phone}`} className="w-6 h-6 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center text-green-600 dark:text-green-400">
                            <span className="material-symbols-outlined text-sm">call</span>
                        </a>
                    </div>
                  </div>

                  <button
                    onClick={() => setShowEditContact(true)}
                    className="w-full mt-2 py-2 border border-primary text-primary rounded-lg text-sm font-bold hover:bg-primary/5 transition-colors"
                  >
                    Update Contact
                  </button>
                </div>
              </details>
            </div>
          )}

          {/* Family / Caretaker Tab */}
          {activeTab === 'Family' && (
              <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 space-y-6">
                  {/* Caretaker Mode Section */}
                  <div>
                      <h3 className="text-gray-900 dark:text-white text-base font-bold mb-3 px-1">I am caring for:</h3>
                      <div className="space-y-3">
                      {familyMembers.map(member => (
                              <div key={member.id} className={`flex items-center justify-between p-4 rounded-xl border transition-all ${activeCaretakerId === member.id ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800' : 'bg-white dark:bg-card-dark border-gray-200 dark:border-slate-800'}`}>
                                  <div className="flex items-center gap-3">
                                      <div className="relative">
                                          <img src={member.avatar} alt={member.name} className="w-12 h-12 rounded-full object-cover" />
                                          <div className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white dark:border-card-dark ${member.status === 'Critical' ? 'bg-red-500' : member.status === 'Warning' ? 'bg-orange-500' : 'bg-green-500'}`}></div>
                                      </div>
                                      <div>
                                          <h4 className="font-bold text-slate-900 dark:text-white">{member.name}</h4>
                                          <p className="text-xs text-slate-500 dark:text-slate-400">{member.relation} • {member.status}</p>
                                      </div>
                                  </div>
                                  <button
                                      onClick={() => toggleCaretakerMode(member)}
                                      className={`px-4 py-2 rounded-lg text-xs font-bold transition-colors ${
                                          activeCaretakerId === member.id
                                          ? 'bg-orange-500 text-white hover:bg-orange-600'
                                          : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600'
                                      }`}
                                  >
                                      {activeCaretakerId === member.id ? 'Exit View' : 'View Profile'}
                                  </button>
                              </div>
                          ))}
                      </div>
                  </div>

                  {/* My Caretakers Section */}
                  <div>
                      <div className="flex justify-between items-center mb-3 px-1">
                          <h3 className="text-gray-900 dark:text-white text-base font-bold">My Caretakers</h3>
                          <button className="text-primary text-xs font-bold flex items-center gap-1 hover:underline">
                              <span className="material-symbols-outlined text-xs">person_add</span> Invite
                          </button>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-800/50 p-4 rounded-xl border border-slate-100 dark:border-slate-700 text-center">
                          <div className="w-12 h-12 bg-slate-200 dark:bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-2">
                              <span className="material-symbols-outlined text-slate-400">family_restroom</span>
                          </div>
                          <p className="text-sm text-slate-500 dark:text-slate-400">No caretakers added yet.</p>
                          <p className="text-xs text-slate-400 mt-1">Invite family members to view your health data in case of emergency.</p>
                      </div>
                  </div>
              </div>
          )}

          {/* Achievements Tab */}
          {activeTab === 'Achievements' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 space-y-4">
              {/* Streak Card */}
              <div className="bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl p-5 text-white shadow-lg relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-white/20 rounded-full -mr-10 -mt-10 blur-2xl"></div>
                <div className="relative z-10 flex items-center gap-4">
                  <div className="w-16 h-16 bg-white/20 backdrop-blur-md rounded-full flex items-center justify-center animate-pulse">
                    <span className="material-symbols-outlined text-4xl filled text-yellow-300">local_fire_department</span>
                  </div>
                  <div>
                    <p className="text-orange-100 text-xs font-bold uppercase tracking-wider">Current Streak</p>
                    <h3 className="text-3xl font-bold">7 Days</h3>
                    <p className="text-sm opacity-90 mt-1">Keep it up! 3 days to next reward.</p>
                  </div>
                </div>
                <div className="mt-4 bg-black/20 rounded-full h-2 w-full overflow-hidden">
                  <div className="bg-yellow-400 h-full rounded-full w-[70%]"></div>
                </div>
              </div>

              {/* Badges Grid */}
              <div className="grid grid-cols-2 gap-3">
                {badgesData.map((badge) => (
                  <div
                    key={badge.id}
                    className={`p-4 rounded-xl border flex flex-col items-center text-center transition-all ${
                      badge.unlocked
                      ? 'bg-white dark:bg-card-dark border-slate-100 dark:border-slate-800 shadow-sm'
                      : 'bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800 opacity-70 grayscale'
                    }`}
                  >
                    <div className={`w-14 h-14 rounded-full flex items-center justify-center mb-3 ${badge.color}`}>
                      <span className="material-symbols-outlined text-2xl">{badge.icon}</span>
                    </div>
                    <h4 className="font-bold text-slate-900 dark:text-white text-sm mb-1">{badge.title}</h4>
                    <p className="text-[10px] text-slate-500 dark:text-slate-400 line-clamp-2 leading-tight">
                      {badge.description}
                    </p>
                    {badge.unlocked ? (
                      <span className="mt-3 text-[10px] font-bold text-green-500 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded-full">
                        Unlocked
                      </span>
                    ) : (
                      <div className="w-full mt-3">
                        <div className="flex justify-between text-[9px] text-slate-400 mb-1">
                          <span>Progress</span>
                          <span>{badge.progress}%</span>
                        </div>
                        <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5">
                          <div className="bg-slate-400 h-full rounded-full" style={{ width: `${badge.progress}%` }}></div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Log Out Button */}
      <div className="p-4 mt-4">
        <button
          onClick={() => navigate('/login')}
          className="flex w-full min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-12 px-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-base font-bold leading-normal tracking-[0.015em] hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
        >
          <span className="truncate">Log Out</span>
        </button>
      </div>

      <div className="text-center py-4 px-4">
        <a className="text-sm text-primary dark:text-primary/90 hover:underline cursor-pointer" onClick={() => navigate('/consent')}>Privacy Policy</a>
      </div>

      {/* --- Modals --- */}
      {showEditPersonal && (
          <EditPersonalModal
            profile={profile}
            onClose={() => setShowEditPersonal(false)}
            onSave={(updated) => { updateProfile(updated); setShowEditPersonal(false); }}
          />
      )}
      {showAddCondition && (
          <AddItemModal
            title="Add Condition"
            placeholder="e.g. Type 2 Diabetes"
            onClose={() => setShowAddCondition(false)}
            onSave={(val) => addItem('conditions', val)}
          />
      )}
      {showAddAllergy && (
          <AddItemModal
            title="Add Allergy"
            placeholder="e.g. Peanuts"
            onClose={() => setShowAddAllergy(false)}
            onSave={(val) => addItem('allergies', val)}
          />
      )}
      {showEditContact && (
          <EditContactModal
            contact={profile.emergencyContact}
            onClose={() => setShowEditContact(false)}
            onSave={(contact) => { updateProfile({...profile, emergencyContact: contact}); setShowEditContact(false); }}
          />
      )}
      {showPhotoModal && (
          <PhotoEditModal
            onClose={() => setShowPhotoModal(false)}
            onSave={(url) => {
              updateProfile({...profile, avatar: url});
              setShowPhotoModal(false);
              // Also sync avatar separately for efficiency
              const userId = user?.id || localStorage.getItem('user_id') || 'default';
              apiClient.updateAvatar(userId, url).catch(err => console.warn('Failed to sync avatar', err));
            }}
          />
      )}

    </div>
  );
};

export default ProfileScreen;
