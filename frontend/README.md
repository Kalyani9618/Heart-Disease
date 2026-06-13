# Cardio AI Assistant - Frontend

Advanced React Native/TypeScript frontend for the Cardiovascular Health Super-App.

## Architecture Overview

```
┌────────────────────┐    ┌────────────────────┐
│   React Frontend   │    │   NLP Service      │
│   (Vite + React)   │◄──►│   (FastAPI)        │
│   Port: 5173       │    │   Port: 5001       │
└────────────────────┘    └────────────────────┘
```

## Overview

Cardio AI Assistant is a comprehensive health management application that helps users monitor their cardiovascular health through AI-powered insights, personalized recommendations, and interactive features.

## Features

### 🏠 Dashboard
- Real-time health metrics visualization
- Daily health insights from AI
- Quick access to all features

### 💬 AI Chat
- Conversational AI assistant for health queries
- Intent recognition and sentiment analysis
- Context-aware responses with medical knowledge

### 💊 Medications
- Medication tracking and reminders
- AI-powered medication insights
- Interaction warnings

### 📅 Appointments
- Schedule and manage appointments
- Provider information
- Appointment reminders

### 🥗 Nutrition
- Meal planning with AI
- Recipe analysis
- Heart-healthy recommendations

### 🏃 Exercise
- Workout tracking
- AI-powered workout analysis
- Personalized exercise recommendations

### 📊 Analytics
- Health trends and patterns
- Risk assessments
- Progress tracking

### 👥 Community
- Connect with others
- Share experiences
- Support groups

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: CSS Modules
- **State Management**: Zustand
- **API Client**: Axios
- **Charts**: Recharts

## Project Structure

```
cardio-ai-assistant/
├── App.tsx                 # Main application component
├── index.tsx               # Entry point
├── package.json            # Dependencies
├── vite.config.ts          # Vite configuration
├── backend/                # Flask backend service
│   ├── aip_service.py      # Main Flask app (port 5000)
│   ├── smart_watch.py      # Smartwatch integration
│   └── ml/                 # ML anomaly detection pipeline
│       ├── alert_pipeline.py
│       ├── anomaly_detector.py
│       ├── chatbot_connector.py
│       ├── feature_extractor.py
│       ├── health_explainer.py
│       ├── prompt_templates.py
│       └── rule_engine.py
├── components/             # Reusable UI components
│   ├── BottomNav.tsx
│   ├── LoadingSpinner.tsx
│   ├── MarkdownRenderer.tsx
│   └── ...
├── screens/                # Page components
│   ├── DashboardScreen.tsx
│   ├── ChatScreen.tsx
│   ├── MedicationScreen.tsx
│   ├── NutritionScreen.tsx
│   ├── ExerciseScreen.tsx
│   ├── AnalyticsDashboard.tsx
│   └── ...
├── services/               # API and external services
│   ├── apiClient.ts        # HTTP client
│   ├── memoryService.ts    # Memory system integration
│   └── ...
├── store/                  # State management (Zustand)
│   ├── useHealthStore.ts
│   ├── useChatStore.ts
│   └── ...
├── contexts/               # React contexts
│   └── LanguageContext.tsx
├── hooks/                  # Custom React hooks
│   ├── useVitals.ts
│   ├── useAppointments.ts
│   └── ...
└── data/                   # Static data and translations
    ├── translations.ts
    ├── recipes.ts
    └── workouts.ts
```

## Quick Start

1. Install dependencies: `npm install`
2. Copy `.env.local.example` to `.env.local` and configure:
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local to set VITE_NLP_SERVICE_URL=http://localhost:5001
   ```
3. Start development server: `npm run dev`
4. Open browser to [http://localhost:5173](http://localhost:5173)

## Development

### Environment Variables

Create a `.env.local` file with these settings:

```bash
# Point to the NLP Service (FastAPI backend)
VITE_NLP_SERVICE_URL=http://localhost:5001

# Optional Gemini API key for direct frontend AI calls
VITE_GEMINI_API_KEY=your-gemini-api-key-here
```

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### Adding New Screens

1. Create component in `screens/`
2. Add route in `App.tsx`
3. Update navigation in `BottomNav.tsx`

### Adding API Endpoints

1. Add endpoint in `backend/aip_service.py`
2. Create service method in `services/apiClient.ts`
3. Use in components via hooks or stores

## Related Services

- **NLP Service**: `../nlp-service/` - Natural language processing
- **Documentation**: `../docs/` - Full project documentation

## License

Part of the HeartGuard project.
