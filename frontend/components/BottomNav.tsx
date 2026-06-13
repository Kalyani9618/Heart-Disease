
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';

const BottomNav: React.FC = () => {
  const location = useLocation();
  const { t } = useLanguage();

  const navItems = [
    { icon: 'home', label: t('nav.home'), path: '/dashboard' },
    { icon: 'monitor_heart', label: t('nav.monitor'), path: '/assessment' },
    { icon: 'chat', label: t('nav.chat'), path: '/chat' },
    { icon: 'calendar_month', label: t('nav.book'), path: '/appointment' },
    { icon: 'settings', label: t('nav.settings'), path: '/settings' },
  ];

  return (
    <div className="bottom-nav-container absolute bottom-0 left-0 right-0 h-[70px] bg-white/95 dark:bg-card-dark/95 backdrop-blur-xl border-t border-slate-200/60 dark:border-slate-800/60 z-50 shadow-[0_-4px_20px_rgba(0,0,0,0.05)] dark:shadow-[0_-4px_20px_rgba(0,0,0,0.2)]">
      <div className="flex justify-around items-center h-full px-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path || location.pathname.startsWith(item.path + '/');
          return (
            <Link
              key={item.label}
              to={item.path}
              data-discover="true"
              className={`relative flex flex-col items-center gap-0.5 min-w-[50px] p-1.5 rounded-xl transition-all duration-300 ${isActive
                  ? 'text-primary'
                  : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
                }`}
            >
              {isActive && (
                <span className="absolute -top-1 left-1/2 -translate-x-1/2 w-5 h-1 bg-primary rounded-full"></span>
              )}
              <span
                className={`material-symbols-outlined text-2xl transition-transform duration-200 ${isActive ? 'filled scale-110' : ''}`}
                style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}
              >
                {item.icon}
              </span>
              <span className={`text-[10px] font-medium ${isActive ? 'font-bold text-primary' : ''}`}>
                {item.label}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
};

export default BottomNav;
