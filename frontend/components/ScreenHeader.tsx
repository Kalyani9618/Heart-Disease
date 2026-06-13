import React from 'react';
import { useNavigate } from 'react-router-dom';

interface ScreenHeaderProps {
    title: string;
    subtitle?: string;
    onRefresh?: () => void;
    rightIcon?: string;
    onRightAction?: () => void;
    className?: string; // Allow custom classes
}

export default function ScreenHeader({
    title,
    subtitle,
    onRefresh,
    rightIcon,
    onRightAction,
    className = ""
}: ScreenHeaderProps) {
    const navigate = useNavigate();

    return (
        <div className={`sticky top-0 z-30 bg-white/80 dark:bg-card-dark/80 backdrop-blur-md border-b border-slate-100 dark:border-slate-800 transition-all ${className}`}>
            <div className="flex items-center p-4">
                <button
                    onClick={() => navigate(-1)}
                    className="p-2 -ml-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 transition-colors active:scale-95"
                    aria-label="Go back"
                >
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>

                <div className="flex-1 text-center mx-2 min-w-0">
                    <h2 className="font-bold text-lg text-slate-900 dark:text-white leading-tight truncate">{title}</h2>
                    {subtitle && (
                        <p className="text-xs text-slate-500 dark:text-slate-400 font-medium truncate">{subtitle}</p>
                    )}
                </div>

                <div className="w-10 flex justify-end shrink-0">
                    {(onRightAction || onRefresh) ? (
                        <button
                            onClick={onRightAction || onRefresh}
                            className="p-2 -mr-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 transition-colors active:scale-95"
                        >
                            <span className="material-symbols-outlined">
                                {rightIcon || (onRefresh ? 'refresh' : 'more_vert')}
                            </span>
                        </button>
                    ) : (
                        <div className="w-10" />
                    )}
                </div>
            </div>
        </div>
    );
}
