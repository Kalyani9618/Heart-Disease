import { useState, useCallback } from 'react';

export interface WaterLog {
  id: string;
  amount: number; // in ml or oz
  timestamp: string;
  notes?: string;
}

export const useWaterTracking = (dailyGoal: number = 2000) => {
  const [waterLogs, setWaterLogs] = useState<WaterLog[]>([]);
  const [loading, setLoading] = useState(false);

  const loadTodaysLogs = useCallback(() => {
    const today = new Date().toDateString();
    const key = `water_logs_${today}`;
    const saved = localStorage.getItem(key);
    if (saved) {
      setWaterLogs(JSON.parse(saved));
    }
  }, []);

  const addWaterLog = useCallback(
    (amount: number) => {
      const newLog: WaterLog = {
        id: `water_${Date.now()}`,
        amount,
        timestamp: new Date().toISOString(),
      };

      const updated = [newLog, ...waterLogs];
      setWaterLogs(updated);

      const today = new Date().toDateString();
      const key = `water_logs_${today}`;
      localStorage.setItem(key, JSON.stringify(updated));

      return newLog;
    },
    [waterLogs]
  );

  const getTotalToday = useCallback((): number => {
    return waterLogs.reduce((sum, log) => sum + log.amount, 0);
  }, [waterLogs]);

  const getProgress = useCallback((): number => {
    const total = getTotalToday();
    return Math.min((total / dailyGoal) * 100, 100);
  }, [getTotalToday, dailyGoal]);

  const getRemainingToday = useCallback((): number => {
    return Math.max(dailyGoal - getTotalToday(), 0);
  }, [getTotalToday, dailyGoal]);

  return {
    waterLogs,
    loading,
    totalToday: getTotalToday(),
    progress: getProgress(),
    remaining: getRemainingToday(),
    addWaterLog,
    loadTodaysLogs,
  };
};
