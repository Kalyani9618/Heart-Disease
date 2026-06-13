
import { Badge } from '../types';

export interface Trophy {
  id: string;
  title: string;
  description: string;
  icon: string;
  tier: 'Gold' | 'Silver' | 'Bronze' | 'Locked';
  dateUnlocked?: string;
}

export const badgesData: Badge[] = [
  {
    id: 'b_1',
    title: 'The Stabilizer',
    description: 'Kept Blood Pressure within healthy range for 7 consecutive days.',
    icon: 'favorite',
    color: 'text-red-500 bg-red-100 dark:bg-red-900/20',
    unlocked: true,
    dateUnlocked: '2023-10-15'
  },
  {
    id: 'b_2',
    title: 'Hydration Hero',
    description: 'Logged 8 glasses of water for 5 days in a row.',
    icon: 'water_drop',
    color: 'text-blue-500 bg-blue-100 dark:bg-blue-900/20',
    unlocked: true,
    dateUnlocked: '2023-10-20'
  },
  {
    id: 'b_3',
    title: 'Sodium Slasher',
    description: 'Stayed under 2000mg sodium daily for a week.',
    icon: 'restaurant',
    color: 'text-green-500 bg-green-100 dark:bg-green-900/20',
    unlocked: false,
    progress: 75
  },
  {
    id: 'b_4',
    title: 'Early Bird',
    description: 'Completed a workout before 8 AM 3 times.',
    icon: 'wb_sunny',
    color: 'text-orange-500 bg-orange-100 dark:bg-orange-900/20',
    unlocked: true,
    dateUnlocked: '2023-10-18'
  },
  {
    id: 'b_5',
    title: 'Weekend Warrior',
    description: 'Hit step goals on Saturday and Sunday.',
    icon: 'hiking',
    color: 'text-purple-500 bg-purple-100 dark:bg-purple-900/20',
    unlocked: false,
    progress: 40
  },
  {
    id: 'b_6',
    title: 'Zen Master',
    description: 'Completed 5 mindfulness or yoga sessions.',
    icon: 'self_improvement',
    color: 'text-teal-500 bg-teal-100 dark:bg-teal-900/20',
    unlocked: false,
    progress: 20
  }
];

export const trophies: Trophy[] = [
    {
        id: 't_1',
        title: 'Marathon Master',
        description: 'Completed 42km total distance.',
        icon: 'emoji_events',
        tier: 'Gold',
        dateUnlocked: '2024-01-15'
    },
    {
        id: 't_2',
        title: 'Iron Heart',
        description: 'Maintained low resting HR for 30 days.',
        icon: 'favorite_border',
        tier: 'Silver',
        dateUnlocked: '2024-02-20'
    },
    {
        id: 't_3',
        title: 'Early Riser',
        description: '10 workouts before 7AM.',
        icon: 'wb_twilight',
        tier: 'Bronze',
        dateUnlocked: '2024-03-10'
    },
    {
        id: 't_4',
        title: 'Century Club',
        description: 'Complete 100 Workouts.',
        icon: 'lock',
        tier: 'Locked'
    }
];
