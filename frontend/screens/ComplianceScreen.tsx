import React, { useState, useEffect } from 'react';
import { apiClient, AuditLogEntry } from '../services/apiClient';
import ScreenHeader from '../components/ScreenHeader';

export default function ComplianceScreen() {
    const [logs, setLogs] = useState<AuditLogEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadAuditLogs();
    }, []);

    const loadAuditLogs = async () => {
        try {
            setLoading(true);
            const data = await apiClient.getAuditLog('current_user', 50);
            setLogs(data);
            setError(null);
        } catch (err) {
            console.error('Failed to load audit logs:', err);
            setError('Failed to load audit logs. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const getActionColor = (action: string) => {
        switch (action) {
            case 'read': return { bg: 'bg-indigo-100 dark:bg-indigo-900/30', text: 'text-indigo-600 dark:text-indigo-400' };
            case 'write': return { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-600 dark:text-emerald-400' };
            case 'delete': return { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-600 dark:text-red-400' };
            case 'export': return { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-600 dark:text-amber-400' };
            default: return { bg: 'bg-slate-100 dark:bg-slate-800', text: 'text-slate-600 dark:text-slate-400' };
        }
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString();
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Compliance Audit Log"
                subtitle="Data Access Records"
                rightIcon="refresh"
                onRightAction={loadAuditLogs}
            />

            <div className="p-4">
                {loading ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading audit records...</p>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <span className="material-symbols-outlined text-5xl text-red-500 mb-3">error</span>
                        <p className="text-red-500 text-sm mb-4">{error}</p>
                        <button
                            onClick={loadAuditLogs}
                            className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-semibold hover:bg-indigo-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : logs.length === 0 ? (
                    <div className="flex flex-col items-center py-20 text-slate-400 dark:text-slate-500">
                        <span className="material-symbols-outlined text-5xl mb-3">fact_check</span>
                        <p className="text-base italic">No audit records found.</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {/* Table Header */}
                        <div className="flex items-center px-3 py-2 border-b border-slate-200 dark:border-slate-700 text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                            <span className="flex-[2]">Timestamp</span>
                            <span className="flex-1">Action</span>
                            <span className="flex-[2]">Resource</span>
                        </div>

                        {/* Log Rows */}
                        {logs.map((log) => {
                            const colors = getActionColor(log.action);
                            return (
                                <div
                                    key={log.id}
                                    className="bg-white dark:bg-card-dark rounded-xl p-3 shadow-sm border border-slate-100 dark:border-slate-800"
                                >
                                    <div className="flex items-center gap-2">
                                        <span className="flex-[2] text-xs text-slate-500 dark:text-slate-400">
                                            {formatDate(log.timestamp)}
                                        </span>
                                        <div className="flex-1">
                                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase ${colors.bg} ${colors.text}`}>
                                                {log.action}
                                            </span>
                                        </div>
                                        <span className="flex-[2] text-sm font-medium text-slate-700 dark:text-slate-200 truncate">
                                            {log.resource_type}
                                        </span>
                                    </div>
                                    {log.details && (
                                        <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-700">
                                            <code className="text-[11px] text-slate-400 dark:text-slate-500 font-mono break-all">
                                                {JSON.stringify(log.details)}
                                            </code>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}
