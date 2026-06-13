import { useState, useEffect } from 'react';
import { apiClient, APIError } from '../services/apiClient';
import { HealthAssessment } from '../types';

export interface DailyInsight {
  text: string;
  timestamp: string;
  loading: boolean;
  error: string | null;
}

export const useDailyInsight = (userName: string, assessment: HealthAssessment | null) => {
  const [insight, setInsight] = useState<DailyInsight>({
    text: '',
    timestamp: '',
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!userName) return;

    const generateInsight = async () => {
      const today = new Date().toDateString();
      const cacheKey = `daily_insight_${today}`;
      const cached = localStorage.getItem(cacheKey);

      // Return cached insight if available
      if (cached) {
        const cachedData = JSON.parse(cached);
        setInsight({
          text: cachedData.text,
          timestamp: cachedData.timestamp,
          loading: false,
          error: null,
        });
        return;
      }

      setInsight((prev) => ({ ...prev, loading: true, error: null }));

      try {
        const response = await apiClient.generateInsight({
          user_name: userName,
          vitals: assessment?.vitals
            ? {
                heart_rate: undefined,
                blood_pressure: `${assessment.vitals.systolic}/80`,
                blood_glucose: undefined,
              }
            : {},
          activities: [],
          medications: [],
        });

        const insightText = response.insight || getDefaultInsight(assessment);
        const timestamp = response.timestamp || new Date().toISOString();

        // Cache the insight
        localStorage.setItem(
          cacheKey,
          JSON.stringify({
            text: insightText,
            timestamp,
            date: today,
          })
        );

        setInsight({
          text: insightText,
          timestamp,
          loading: false,
          error: null,
        });
      } catch (error) {
        const errorMessage =
          error instanceof APIError ? error.message : 'Failed to generate insight';
        setInsight({
          text: getDefaultInsight(assessment),
          timestamp: new Date().toISOString(),
          loading: false,
          error: errorMessage,
        });
      }
    };

    generateInsight();
  }, [userName, assessment]);

  return insight;
};

const getDefaultInsight = (assessment: HealthAssessment | null): string => {
  if (!assessment) {
    return 'Complete your assessment to receive personalized AI health insights about your heart health trends.';
  }
  if (assessment.risk === 'High Risk') {
    return 'Your recent assessment indicates potential high risk factors. It is recommended to consult a specialist soon.';
  }
  if (assessment.risk === 'Moderate Risk') {
    return 'Focusing on a heart-healthy lifestyle and increasing daily steps can help improve your score.';
  }
  return 'Great job! Your latest assessment shows low risk. Maintenance is key—keep up your current healthy habits.';
};
