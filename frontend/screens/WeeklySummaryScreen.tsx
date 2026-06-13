import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/apiClient';
import ScreenHeader from '../components/ScreenHeader';

export default function WeeklySummaryScreen() {
    const [loading, setLoading] = useState(true);
    const [summary, setSummary] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadSummary();
    }, []);

    const loadSummary = async () => {
        try {
            setLoading(true);
            const data = await apiClient.getWeeklySummary('current_user');
            setSummary(data);
            setError(null);
        } catch (err) {
            console.error('Failed to load weekly summary:', err);
            setError('Failed to load weekly summary. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    const StatCard = ({ icon, title, value, subtitle, color = 'indigo' }: { icon: string; title: string; value: string | number; subtitle?: string; color?: string }) => {
        const colorMap: Record<string, { bg: string; text: string }> = {
            indigo: { bg: 'bg-indigo-100 dark:bg-indigo-900/30', text: 'text-indigo-600 dark:text-indigo-400' },
            pink: { bg: 'bg-pink-100 dark:bg-pink-900/30', text: 'text-pink-600 dark:text-pink-400' },
            blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-600 dark:text-blue-400' },
            green: { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-600 dark:text-emerald-400' },
            orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-600 dark:text-orange-400' },
            purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', text: 'text-purple-600 dark:text-purple-400' },
            red: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-600 dark:text-red-400' },
        };
        const c = colorMap[color] || colorMap.indigo;
        return (
            <div className="flex items-center gap-3 bg-white dark:bg-card-dark rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-800">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 ${c.bg}`}>
                    <span className={`material-symbols-outlined text-2xl ${c.text}`}>{icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-500 dark:text-slate-400">{title}</p>
                    <p className="text-xl font-bold text-slate-800 dark:text-white">{value}</p>
                    {subtitle && <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">{subtitle}</p>}
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Weekly Summary"
                subtitle="Your Health Overview"
                rightIcon="refresh"
                onRightAction={loadSummary}
            />

            <div className="p-4 space-y-4">
                {loading ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading summary...</p>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <span className="material-symbols-outlined text-5xl text-red-500 mb-3">error</span>
                        <p className="text-red-500 text-sm mb-4">{error}</p>
                        <button
                            onClick={loadSummary}
                            className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-semibold hover:bg-indigo-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : summary ? (
                    <>
                        {/* Period Card */}
                        <div className="bg-white dark:bg-card-dark rounded-xl p-4 border-l-4 border-indigo-500 shadow-sm">
                            <p className="text-[11px] text-slate-500 dark:text-slate-400 uppercase font-semibold tracking-wider mb-1">Summary Period</p>
                            <p className="text-base font-medium text-slate-800 dark:text-white">
                                {formatDate(summary.week_start)} — {formatDate(summary.week_end)}
                            </p>
                        </div>

                        {/* Personalized Tip */}
                        {summary.personalized_tip && (
                            <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-4 border-l-4 border-amber-400">
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="material-symbols-outlined text-lg text-amber-500 filled">lightbulb</span>
                                    <h4 className="text-sm font-bold text-slate-800 dark:text-white">Personalized Tip</h4>
                                </div>
                                <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{summary.personalized_tip}</p>
                            </div>
                        )}

                        {/* Health Stats */}
                        <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Health Stats</h2>
                        <div className="space-y-2.5">
                            <StatCard icon="favorite" title="Avg Heart Rate" value={`${summary.health_stats?.avg_heart_rate || 0} BPM`} color="pink" />
                            <StatCard icon="directions_walk" title="Total Steps" value={(summary.health_stats?.total_steps || 0).toLocaleString()} subtitle={`${summary.health_stats?.avg_steps_per_day || 0}/day avg`} color="blue" />
                            <StatCard icon="emoji_events" title="Goal Met" value={`${summary.health_stats?.steps_goal_met_days || 0} days`} color="green" />
                        </div>

                        {/* Medications */}
                        <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Medications</h2>
                        <div className="bg-white dark:bg-card-dark rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-800">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">Overall Compliance</span>
                                <span className={`text-2xl font-bold ${
                                    (summary.medications?.overall_compliance_percent || 0) >= 80 ? 'text-emerald-500' : 'text-amber-500'
                                }`}>
                                    {Math.round(summary.medications?.overall_compliance_percent || 0)}%
                                </span>
                            </div>
                            <p className="text-sm text-slate-500 dark:text-slate-400">
                                {summary.medications?.total_doses_taken || 0} taken, {summary.medications?.total_doses_missed || 0} missed
                            </p>
                        </div>

                        {/* Highlights */}
                        {summary.highlights && summary.highlights.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Highlights</h2>
                                <div className="space-y-2">
                                    {summary.highlights.map((highlight: string, index: number) => (
                                        <div key={index} className="flex items-start gap-2.5 bg-white dark:bg-card-dark rounded-xl p-3 shadow-sm border border-slate-100 dark:border-slate-800">
                                            <span className="material-symbols-outlined text-base text-emerald-500 mt-0.5 filled">check_circle</span>
                                            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed flex-1">{highlight}</p>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}

                        {/* Areas for Improvement */}
                        {summary.areas_for_improvement && summary.areas_for_improvement.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Areas for Improvement</h2>
                                <div className="space-y-2">
                                    {summary.areas_for_improvement.map((area: string, index: number) => (
                                        <div key={index} className="flex items-start gap-2.5 bg-white dark:bg-card-dark rounded-xl p-3 shadow-sm border border-slate-100 dark:border-slate-800">
                                            <span className="material-symbols-outlined text-base text-amber-500 mt-0.5">trending_up</span>
                                            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed flex-1">{area}</p>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col items-center py-20 text-slate-400 dark:text-slate-500">
                        <span className="material-symbols-outlined text-5xl mb-3">summarize</span>
                        <p className="text-base italic">No summary data available</p>
                    </div>
                )}
            </div>
        </div>
    );
}
