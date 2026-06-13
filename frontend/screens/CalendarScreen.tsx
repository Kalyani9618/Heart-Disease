import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/apiClient';
import { useAuth } from '../hooks/useAuth';
import { useToast } from '../components/Toast';
import ScreenHeader from '../components/ScreenHeader';

export default function CalendarScreen() {
    const { user } = useAuth();
    const { showToast } = useToast();
    const [loading, setLoading] = useState(false);
    const [events, setEvents] = useState<any[]>([]);
    const [syncing, setSyncing] = useState(false);

    useEffect(() => {
        loadEvents();
    }, []);

    const loadEvents = async () => {
        if (!user) return;
        setLoading(true);
        try {
            const data = await apiClient.getCalendarEvents(user.id);
            setEvents(data);
        } catch (error) {
            console.error('Load events error:', error);
            showToast('Failed to load calendar events', 'error');
            setEvents([]);
        } finally {
            setLoading(false);
        }
    };

    const handleSync = async () => {
        if (!user) return;
        setSyncing(true);
        try {
            await apiClient.syncCalendar(user.id, { provider: 'google' });
            showToast('Calendar synced successfully', 'success');
            loadEvents();
        } catch (error) {
            console.error('Sync error:', error);
            showToast('Failed to sync calendar', 'error');
        } finally {
            setSyncing(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Calendar"
                subtitle="Events & Schedule"
                rightIcon="sync"
                onRightAction={handleSync}
            />

            <div className="p-4 space-y-5">
                {/* Summary Card */}
                <div className="bg-gradient-to-br from-indigo-600 to-violet-500 rounded-2xl p-5 text-white shadow-lg shadow-indigo-500/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-40 h-40 bg-white/10 rounded-full blur-3xl -mr-10 -mt-10"></div>
                    <div className="relative z-10 flex items-center justify-between">
                        <div>
                            <h3 className="text-lg font-bold">Upcoming Events</h3>
                            <p className="text-white/80 text-sm mt-1">{events.length} events this week</p>
                        </div>
                        <span className="material-symbols-outlined text-4xl text-white/70">calendar_month</span>
                    </div>
                </div>

                {/* Section Title */}
                <h2 className="text-lg font-bold text-slate-800 dark:text-white">Schedule</h2>

                {/* Event List */}
                {loading ? (
                    <div className="flex justify-center py-16">
                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                ) : events.length === 0 ? (
                    <div className="flex flex-col items-center py-16 text-slate-400 dark:text-slate-500">
                        <span className="material-symbols-outlined text-5xl mb-3">event_busy</span>
                        <p className="text-base">No upcoming events</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {events.map((item: any) => (
                            <div
                                key={item.id}
                                className="flex bg-white dark:bg-card-dark rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-800"
                            >
                                {/* Time Column */}
                                <div className="border-r border-slate-200 dark:border-slate-700 pr-4 mr-4 flex flex-col items-center justify-center min-w-[70px]">
                                    <span className="text-base font-bold text-slate-800 dark:text-white">
                                        {new Date(item.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                    <span className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                                        {new Date(item.start_time).toLocaleDateString()}
                                    </span>
                                </div>
                                {/* Details Column */}
                                <div className="flex-1 flex flex-col justify-center min-w-0">
                                    <h4 className="text-base font-semibold text-slate-800 dark:text-white truncate">
                                        {item.title}
                                    </h4>
                                    {item.location && (
                                        <div className="flex items-center gap-1 mt-1">
                                            <span className="material-symbols-outlined text-sm text-slate-400">location_on</span>
                                            <span className="text-sm text-slate-500 dark:text-slate-400 truncate">{item.location}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
