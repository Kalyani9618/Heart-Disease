"""
Semantic Router v2 - Production-Ready Intent Classification

This is the Phase 1.1 implementation of the Unified Master Plan.
Routes medical queries to the appropriate handler (SQL, RAG, Emergency, etc).

Key Design:
- Regex first (safety-critical, <10ms)
- Embedding fallback (optional, for edge cases)
- Medical triage patterns optimized for emergency detection
- Confidence scoring for decision transparency

Integration:
- routes/orchestrated_chat.py uses this router
- agents/langgraph_orchestrator.py selects node based on intent
"""


import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Top-level intent categories for Heart Health AI."""
    EMERGENCY = "emergency"              # "chest pain", "911", "can't breathe"
    HEART_RISK = "heart_risk"            # "heart disease risk", "am I having a heart attack?"
    VITALS_QUERY = "vitals_query"       # "What is my heart rate?", "BP trends"
    MEDICAL_QA = "medical_qa"           # "What causes hypertension?", "Drug info"
    DRUG_INTERACTION = "drug_interaction" # "Can I take aspirin with warfarin?"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis" # "What could be causing my symptoms?"
    TRIAGE = "triage"                 # "Should I go to the ER?"
    RESEARCH = "research"               # "Research latest cardiology studies"
    GENERAL = "general"                 # "Hi", "Thanks", off-topic


@dataclass
class RouteDecision:
    """Result of semantic routing."""
    intent: IntentCategory
    confidence: float  # 0.0-1.0
    target_handler: str  # "emergency_handler", "sql_tool", "rag_tool", etc.
    reasoning: str
    requires_embedding: bool = False
    matched_pattern: Optional[str] = None


class SemanticRouterV2:
    """
    Routes medical queries to appropriate handler.
    
    Pattern: Regex first (safety-critical), embedding second (optional accuracy).
    
    Example Usage:
        router = SemanticRouterV2()
        
        # Emergency detection
        decision = router.route("I can't breathe")
        assert decision.intent == IntentCategory.EMERGENCY
        assert decision.confidence == 0.95
        
        # Vitals query
        decision = router.route("What was my heart rate last week?")
        assert decision.intent == IntentCategory.VITALS_QUERY
        assert decision.target_handler == "sql_tool"
        
        # Medical Q&A
        decision = router.route("What are the side effects of Lisinopril?")
        assert decision.intent == IntentCategory.MEDICAL_QA
        assert decision.target_handler == "rag_tool"
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1: EMERGENCY DETECTION (Regex, Safety-Critical, <10ms)
    # ═══════════════════════════════════════════════════════════════════════════
    
    EMERGENCY_PATTERNS = [
        # Cardiac emergencies (English)
        (r"\b(chest\s+pain|chest\s+tightness|chest\s+pressure)\b", 0.95, "Chest pain detected"),
        (r"\b(can't\s+breathe|cannot\s+breathe|difficulty\s+breathing|shortness\s+of\s+breath|dyspnea)\b", 0.95, "Breathing difficulty detected"),
        (r"\b(heart\s+attack|myocardial\s+infarction|mi|cardiac\s+arrest)\b", 0.95, "Heart attack detected"),
        (r"\b(severe\s+chest|critical\s+heart)\b", 0.95, "Severe cardiac symptom detected"),
        
        # Stroke/Neurological (English)
        (r"\b(stroke|tia|transient\s+ischemic\s+attack|paralysis|weakness|slurred\s+speech)\b", 0.95, "Stroke symptoms detected"),
        (r"\b(loss\s+of\s+(consciousness|vision)|fainting|syncope)\b", 0.90, "Loss of consciousness detected"),
        
        # Emergency keywords (English)
        (r"\b(911|call\s+911|emergency|urgent\s+help|crisis|immediate\s+help)\b", 0.90, "Emergency keyword detected"),
        
        # Severe bleeding (English)
        (r"\b(severe\s+bleeding|hemorrhage|uncontrolled\s+bleeding)\b", 0.90, "Severe bleeding detected"),
        
        # Severe pain (English)
        (r"\b(severe\s+pain|unbearable\s+pain|excruciating)\b", 0.85, "Severe pain reported"),
        
        # Unconscious/collapsed (English)
        (r"\b(unconscious|collapsed|unresponsive|not\s+responding)\b", 0.90, "Unconscious/collapsed detected"),
        
        # ─── SPANISH (ES) EMERGENCIES ───
        (r"\b(dolor\s+(de\s+)?pecho|opresión\s+(en\s+el\s+)?pecho|presión\s+(en\s+el\s+)?pecho)\b", 0.95, "ES: Dolor de pecho detectado"),
        (r"\b(no\s+puedo\s+respirar|dificultad\s+(para\s+)?respirar|falta\s+de\s+aire|ahogamiento)\b", 0.95, "ES: Dificultad respiratoria detectada"),
        (r"\b(ataque\s+(al\s+)?corazón|infarto|paro\s+cardíaco|infarto\s+de\s+miocardio)\b", 0.95, "ES: Ataque al corazón detectado"),
        (r"\b(derrame\s+cerebral|accidente\s+cerebrovascular|ictus|parálisis|debilidad)\b", 0.95, "ES: Síntomas de derrame detectados"),
        (r"\b(pérdida\s+de\s+(conocimiento|visión)|desmayo|síncope)\b", 0.90, "ES: Pérdida de conocimiento detectada"),
        (r"\b(emergencia|urgencia|ayuda\s+urgente|llamar\s+(al\s+)?911)\b", 0.90, "ES: Palabra de emergencia detectada"),
        (r"\b(hemorragia|sangrado\s+(severo|grave))\b", 0.90, "ES: Sangrado severo detectado"),
        (r"\b(dolor\s+(severo|insoportable)|agonía)\b", 0.85, "ES: Dolor severo reportado"),
        
        # ─── FRENCH (FR) EMERGENCIES ───
        (r"\b(douleur\s+(à\s+la\s+)?poitrine|oppression\s+(thoracique|poitrine))\b", 0.95, "FR: Douleur thoracique détectée"),
        (r"\b(je\s+ne\s+peux\s+pas\s+respirer|difficulté\s+à\s+respirer|essoufflement|dyspnée)\b", 0.95, "FR: Difficulté respiratoire détectée"),
        (r"\b(crise\s+cardiaque|infarctus|arrêt\s+cardiaque)\b", 0.95, "FR: Crise cardiaque détectée"),
        (r"\b(avc|accident\s+vasculaire\s+cérébral|attaque\s+cérébrale|paralysie|faiblesse)\b", 0.95, "FR: Symptômes d'AVC détectés"),
        (r"\b(perte\s+de\s+(connaissance|vision)|évanouissement|syncope)\b", 0.90, "FR: Perte de connaissance détectée"),
        (r"\b(urgence|aide\s+urgente|appeler\s+(le\s+)?15|samu)\b", 0.90, "FR: Mot d'urgence détecté"),
        (r"\b(hémorragie|saignement\s+(sévère|grave))\b", 0.90, "FR: Saignement sévère détecté"),
        (r"\b(douleur\s+(sévère|insupportable)|atroce)\b", 0.85, "FR: Douleur sévère signalée"),
        
        # ─── GERMAN (DE) EMERGENCIES ───
        (r"\b(brustschmerzen|brustenge|druck\s+(auf\s+der\s+)?brust)\b", 0.95, "DE: Brustschmerzen erkannt"),
        (r"\b(kann\s+nicht\s+atmen|atemnot|kurzatmigkeit|dyspnoe)\b", 0.95, "DE: Atemnot erkannt"),
        (r"\b(herzinfarkt|herzanfall|herzstillstand|myokardinfarkt)\b", 0.95, "DE: Herzinfarkt erkannt"),
        (r"\b(schlaganfall|hirnschlag|lähmung|schwäche|sprachstörung)\b", 0.95, "DE: Schlaganfall-Symptome erkannt"),
        (r"\b(bewusstlosigkeit|ohnmacht|synkope|sehverlust)\b", 0.90, "DE: Bewusstlosigkeit erkannt"),
        (r"\b(notfall|notruf|hilfe|112\s+anrufen|rettungsdienst)\b", 0.90, "DE: Notfall-Schlüsselwort erkannt"),
        (r"\b(starke\s+blutung|hämorrhagie|blutung)\b", 0.90, "DE: Schwere Blutung erkannt"),
        (r"\b(starke\s+schmerzen|unerträgliche\s+schmerzen)\b", 0.85, "DE: Starke Schmerzen gemeldet"),
        
        # ─── PORTUGUESE (PT) EMERGENCIES ───
        (r"\b(dor\s+no\s+peito|aperto\s+no\s+peito|pressão\s+no\s+peito)\b", 0.95, "PT: Dor no peito detectada"),
        (r"\b(não\s+consigo\s+respirar|falta\s+de\s+ar|dificuldade\s+(para\s+)?respirar)\b", 0.95, "PT: Dificuldade respiratória detectada"),
        (r"\b(ataque\s+cardíaco|infarto|parada\s+cardíaca)\b", 0.95, "PT: Ataque cardíaco detectado"),
        (r"\b(avc|derrame|paralisia|fraqueza)\b", 0.95, "PT: Sintomas de AVC detectados"),
        (r"\b(emergência|urgência|chamar\s+(o\s+)?192|samu)\b", 0.90, "PT: Palavra de emergência detectada"),
        
        # ─── ITALIAN (IT) EMERGENCIES ───
        (r"\b(dolore\s+al\s+petto|oppressione\s+toracica|pressione\s+al\s+petto)\b", 0.95, "IT: Dolore toracico rilevato"),
        (r"\b(non\s+riesco\s+a\s+respirare|difficoltà\s+a\s+respirare|mancanza\s+di\s+respiro)\b", 0.95, "IT: Difficoltà respiratoria rilevata"),
        (r"\b(infarto|attacco\s+di\s+cuore|arresto\s+cardiaco)\b", 0.95, "IT: Infarto rilevato"),
        (r"\b(ictus|paralisi|debolezza|parla\s+male)\b", 0.95, "IT: Sintomi ictus rilevati"),
        (r"\b(emergenza|urgenza|chiamare\s+(il\s+)?118)\b", 0.90, "IT: Parola di emergenza rilevata"),
        
        # ─── CHINESE (ZH) EMERGENCIES ───
        (r"(胸痛|胸闷|胸部压迫感|心绞痛)", 0.95, "ZH: 胸痛检测"),
        (r"(无法呼吸|呼吸困难|喘不上气|气短)", 0.95, "ZH: 呼吸困难检测"),
        (r"(心脏病发作|心肌梗死|心脏骤停|心梗)", 0.95, "ZH: 心脏病发作检测"),
        (r"(中风|脑卒中|瘫痪|麻木|言语不清)", 0.95, "ZH: 中风症状检测"),
        (r"(紧急情况|急救|拨打120|救命)", 0.90, "ZH: 紧急关键词检测"),
        
        # ─── JAPANESE (JA) EMERGENCIES ───
        (r"(胸の痛み|胸が苦しい|胸が締め付けられる)", 0.95, "JA: 胸痛検出"),
        (r"(息ができない|呼吸困難|息切れ)", 0.95, "JA: 呼吸困難検出"),
        (r"(心臓発作|心筋梗塞|心停止)", 0.95, "JA: 心臓発作検出"),
        (r"(脳卒中|麻痺|しびれ|ろれつが回らない)", 0.95, "JA: 脳卒中症状検出"),
        (r"(緊急|救急|119番|助けて)", 0.90, "JA: 緊急キーワード検出"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2: VITALS QUERY DETECTION (Regex, High Confidence, <10ms)
    # ═══════════════════════════════════════════════════════════════════════════
    
    VITALS_PATTERNS = [
        # Heart rate queries
        (r"\b(heart\s+rate|pulse|hr|bpm|beats?\s+per\s+minute)\b", 0.90, "Heart rate query"),
        (r"\b(what.*?\bheart\s+rate|show.*?\bhr|my.*?\bpulse)\b", 0.85, "Heart rate lookup"),
        
        # Blood pressure queries
        (r"\b(blood\s+pressure|bp|systolic|diastolic)\b", 0.90, "Blood pressure query"),
        (r"\b(what.*?\bblood\s+pressure|show.*?\bbp|my.*?\bblood\s+pressure)\b", 0.85, "Blood pressure lookup"),
        
        # Generic metrics/trends
        (r"\b(metrics?|vital\s+signs|readings?|values?)\b", 0.80, "Vitals metrics query"),
        (r"\b(trend|history|last\s+(week|month|day|7\s+days|30\s+days))\b", 0.75, "Historical vitals"),
        
        # Data retrieval patterns
        (r"\b(what\s+(was|is|are).*?(reading|trend|value|history))\b", 0.80, "Data retrieval pattern"),
        (r"\b(show|display|list).*?(vitals|metrics|readings|health\s+data)\b", 0.85, "Explicit vitals request"),
        
        # Exercise/activity tracking
        (r"\b(during.*?exercise|during.*?workout|during.*?run|heart\s+rate.*?exercise)\b", 0.85, "Exercise metrics"),
        (r"\b(rest|resting|sleep|sleeping)\s+(heart\s+rate|pulse|hr)\b", 0.85, "Rest metrics"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2.5: HEART RISK ASSESSMENT (RAG-augmented prediction)
    # ═══════════════════════════════════════════════════════════════════════════
    
    HEART_RISK_PATTERNS = [
        # Risk assessment queries
        (r"\b(heart\s+disease\s+risk|risk\s+of\s+heart|heart.*risk\s+assessment)\b", 0.90, "Heart disease risk query"),
        (r"\b(am\s+i\s+having\s+a\s+heart\s+attack|could\s+this\s+be\s+a\s+heart\s+attack)\b", 0.92, "Heart attack inquiry"),
        (r"\b(assess.*?heart|evaluate.*?cardiac|check.*?heart\s+health)\b", 0.85, "Heart health assessment"),
        
        # Atypical symptom patterns (esp. for women)
        (r"\b(jaw\s+pain|fatigue.*nausea|nausea.*fatigue)\b.*\b(heart|cardiac)?\b", 0.80, "Atypical heart symptoms"),
        (r"\b(woman|female)\b.*\b(heart|cardiac|chest|symptom)\b", 0.80, "Female cardiac query"),
        
        # Cardiovascular symptom patterns  
        (r"\b(angina|palpitation|arrhythmia|heart\s+flutter|irregular\s+heartbeat)\b", 0.85, "Cardiac symptom"),
        (r"\b(myocardial|infarction|cardiomyopathy|heart\s+failure)\b", 0.85, "Cardiac condition"),
        (r"\b(coronary)\b.*\b(risk|assess|predict|evaluat|check)\b", 0.85, "Coronary risk assessment"),
        
        # Risk factor queries
        (r"\b(cardiovascular\s+risk|cardiac\s+risk|cholesterol.*heart|ldl.*risk)\b", 0.85, "CV risk factors"),
        (r"\b(predict.*heart|likelihood.*heart|chance.*cardiac)\b", 0.80, "Heart prediction query"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 3: TRIAGE & DIFFERENTIAL DIAGNOSIS (Clinical Reasoning)
    # ═══════════════════════════════════════════════════════════════════════════

    TRIAGE_PATTERNS = [
        # English triage patterns
        (r"\b(should\s+i\s+go\s+to\s+the\s+(er|emergency|hospital))\b", 0.90, "ER Triage query"),
        (r"\b(do\s+i\s+need\s+to\s+see\s+a\s+doctor)\b", 0.85, "Doctor visit query"),
        (r"\b(is\s+this\s+(serious|urgent|emergency))\b", 0.85, "Urgency assessment"),
        (r"\b(triage|assess\s+my\s+symptoms)\b", 0.85, "Explicit triage request"),
        
        # ─── SPANISH (ES) TRIAGE ───
        (r"\b(debo\s+ir\s+al\s+(hospital|urgencias|emergencias))\b", 0.90, "ES: Consulta de triaje"),
        (r"\b(necesito\s+ver\s+a\s+un\s+(médico|doctor))\b", 0.85, "ES: Consulta médica"),
        (r"\b(es\s+(grave|urgente|serio))\b", 0.85, "ES: Evaluación de urgencia"),
        
        # ─── FRENCH (FR) TRIAGE ───
        (r"\b(dois-je\s+aller\s+aux\s+(urgences|hôpital))\b", 0.90, "FR: Question de triage"),
        (r"\b(dois-je\s+voir\s+un\s+(médecin|docteur))\b", 0.85, "FR: Consultation médicale"),
        (r"\b(est-ce\s+(grave|urgent|sérieux))\b", 0.85, "FR: Évaluation d'urgence"),
        
        # ─── GERMAN (DE) TRIAGE ───
        (r"\b(soll\s+ich\s+(ins\s+krankenhaus|in\s+die\s+notaufnahme))\b", 0.90, "DE: Triage-Anfrage"),
        (r"\b(muss\s+ich\s+zum\s+arzt)\b", 0.85, "DE: Arztbesuch-Anfrage"),
        (r"\b(ist\s+(es|das)\s+(ernst|dringend|gefährlich))\b", 0.85, "DE: Dringlichkeitsbewertung"),
        
        # ─── PORTUGUESE (PT) TRIAGE ───
        (r"\b(devo\s+ir\s+ao\s+(hospital|pronto-socorro))\b", 0.90, "PT: Consulta de triagem"),
        (r"\b(preciso\s+ver\s+um\s+(médico|doutor))\b", 0.85, "PT: Consulta médica"),
        
        # ─── ITALIAN (IT) TRIAGE ───
        (r"\b(devo\s+andare\s+al\s+(pronto\s+soccorso|ospedale))\b", 0.90, "IT: Domanda di triage"),
        (r"\b(devo\s+vedere\s+un\s+(medico|dottore))\b", 0.85, "IT: Visita medica"),
    ]

    DIFFERENTIAL_DIAGNOSIS_PATTERNS = [
        # English
        (r"\b(what\s+(could\s+be|is)\s+causing\s+my\s+symptoms)\b", 0.85, "Causality query"),
        (r"\b(what\s+do\s+i\s+have)\b", 0.80, "Diagnosis query"),
        (r"\b(diagnose\s+me|differential\s+diagnosis)\b", 0.85, "Explicit diagnosis request"),
        (r"\b(possible\s+causes\s+for)\b", 0.80, "Causes query"),
        (r"\b(could\s+it\s+be)\b", 0.75, "Hypothesis testing"),
        
        # ─── SPANISH (ES) DIAGNOSIS ───
        (r"\b(qué\s+(puede\s+ser|está\s+causando)\s+mis\s+síntomas)\b", 0.85, "ES: Consulta de causalidad"),
        (r"\b(qué\s+tengo|cuál\s+es\s+mi\s+diagnóstico)\b", 0.80, "ES: Consulta de diagnóstico"),
        (r"\b(causas\s+posibles)\b", 0.80, "ES: Consulta de causas"),
        
        # ─── FRENCH (FR) DIAGNOSIS ───
        (r"\b(qu'est-ce\s+qui\s+(cause|provoque)\s+mes\s+symptômes)\b", 0.85, "FR: Question de causalité"),
        (r"\b(qu'est-ce\s+que\s+j'ai)\b", 0.80, "FR: Question de diagnostic"),
        (r"\b(causes\s+possibles)\b", 0.80, "FR: Question de causes"),
        
        # ─── GERMAN (DE) DIAGNOSIS ───
        (r"\b(was\s+verursacht\s+meine\s+symptome)\b", 0.85, "DE: Kausalitätsanfrage"),
        (r"\b(was\s+habe\s+ich)\b", 0.80, "DE: Diagnoseanfrage"),
        (r"\b(mögliche\s+ursachen)\b", 0.80, "DE: Ursachenanfrage"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 4: MEDICAL QA DETECTION (Regex, Lower Confidence)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MEDICAL_QA_PATTERNS = [
        # English Question patterns
        (r"\b(what\s+are?|why|how|which)\b.*\b(symptom|cause|treatment|medication|drug|side.*effect)\b", 0.80, "Medical question pattern"),
        
        # "What is X used for?" or "What is X for?" patterns (common medical questions)
        (r"\bwhat\s+is\b.*\b(used\s+for|for\s+treating|prescribed\s+for|indicated\s+for)\b", 0.85, "Drug purpose question"),
        
        # Specific conditions
        (r"\b(hypertension|high\s+blood\s+pressure|diabetes|arrhythmia|atrial\s+fibrillation|heart\s+disease|coronary|myocardial)\b", 0.85, "Medical condition"),
        
        # Extended list of common medications (cardiovascular, diabetes, common prescriptions)
        (r"\b(lisinopril|enalapril|metoprolol|atenolol|amlodipine|losartan|warfarin|aspirin|clopidogrel)\b", 0.90, "Specific drug query"),
        (r"\b(metformin|glipizide|glyburide|sitagliptin|jardiance|januvia|ozempic|trulicity|insulin)\b", 0.90, "Diabetes medication"),
        (r"\b(atorvastatin|simvastatin|rosuvastatin|pravastatin|lipitor|crestor|zocor)\b", 0.90, "Statin medication"),
        (r"\b(omeprazole|pantoprazole|ranitidine|famotidine|nexium|prilosec)\b", 0.90, "GI medication"),
        (r"\b(levothyroxine|synthroid|armour|cytomel)\b", 0.90, "Thyroid medication"),
        (r"\b(prednisone|methylprednisolone|dexamethasone|hydrocortisone)\b", 0.90, "Steroid medication"),
        (r"\b(gabapentin|pregabalin|lyrica|neurontin)\b", 0.90, "Nerve medication"),
        (r"\b(sertraline|escitalopram|fluoxetine|paroxetine|citalopram|zoloft|lexapro|prozac)\b", 0.90, "Antidepressant medication"),
        (r"\b(alprazolam|lorazepam|diazepam|clonazepam|xanax|ativan|valium|klonopin)\b", 0.90, "Anxiolytic medication"),
        (r"\b(hydrochlorothiazide|furosemide|spironolactone|chlorthalidone|lasix)\b", 0.90, "Diuretic medication"),
        
        # Generic medication pattern - ends with common drug suffixes
        (r"\b\w+(pril|sartan|olol|dipine|statin|prazole|mab|nib|zole|mycin|cillin|cycline)\b", 0.85, "Drug name suffix detected"),
        
        # General medication topic
        (r"\b(medication|drug|medicine|pharmaceutical|prescription)\b", 0.75, "General medication topic"),
        
        # Medical symptoms
        (r"\b(symptom|symptomatology|presents?\s+with|presenting)\b", 0.80, "Symptom discussion"),
        (r"\b(side\s+effect|adverse|reaction|contraindication|interaction)\b", 0.85, "Drug side effects"),
        
        # Guidelines/treatment
        (r"\b(guideline|treatment|therapy|management|dosage|dose)\b", 0.75, "Treatment guidelines"),
        (r"\b(how\s+to.*treat|what.*treatment|what.*medication)\b", 0.80, "Treatment question"),
        
        # Health conditions (expanded)
        (r"\b(disease|disorder|condition|syndrome|illness|ailment)\b", 0.75, "Health condition topic"),
        (r"\b(diagnos|prognos|therap|patholog|physiolog)\w*\b", 0.80, "Medical terminology"),
        
        # ─── SPANISH (ES) MEDICAL QA ───
        (r"\b(cuáles\s+son|por\s+qué|cómo|qué)\b.*\b(síntoma|causa|tratamiento|medicamento|medicación|efecto)\b", 0.80, "ES: Patrón de pregunta médica"),
        (r"\b(para\s+qué\s+sirve|para\s+qué\s+se\s+usa)\b", 0.85, "ES: Pregunta de propósito del medicamento"),
        (r"\b(hipertensión|presión\s+arterial\s+alta|diabetes|arritmia|enfermedad\s+cardíaca)\b", 0.85, "ES: Condición médica"),
        (r"\b(medicamento|medicina|fármaco|receta)\b", 0.75, "ES: Tema de medicación"),
        (r"\b(síntoma|presenta\s+con)\b", 0.80, "ES: Discusión de síntomas"),
        (r"\b(efecto\s+secundario|reacción\s+adversa|contraindicación)\b", 0.85, "ES: Efectos secundarios"),
        (r"\b(tratamiento|terapia|dosis|dosificación)\b", 0.75, "ES: Guías de tratamiento"),
        (r"\b(enfermedad|trastorno|condición|síndrome)\b", 0.75, "ES: Tema de salud"),
        
        # ─── FRENCH (FR) MEDICAL QA ───
        (r"\b(quels\s+sont|pourquoi|comment|quel)\b.*\b(symptôme|cause|traitement|médicament|effet)\b", 0.80, "FR: Modèle de question médicale"),
        (r"\b(à\s+quoi\s+sert|pour\s+quoi\s+est\s+utilisé)\b", 0.85, "FR: Question sur l'usage du médicament"),
        (r"\b(hypertension|tension\s+artérielle\s+élevée|diabète|arythmie|maladie\s+cardiaque)\b", 0.85, "FR: Condition médicale"),
        (r"\b(médicament|remède|ordonnance)\b", 0.75, "FR: Sujet de médication"),
        (r"\b(symptôme|se\s+présente\s+avec)\b", 0.80, "FR: Discussion de symptômes"),
        (r"\b(effet\s+secondaire|réaction\s+indésirable|contre-indication)\b", 0.85, "FR: Effets secondaires"),
        (r"\b(traitement|thérapie|dosage|posologie)\b", 0.75, "FR: Directives de traitement"),
        (r"\b(maladie|trouble|syndrome)\b", 0.75, "FR: Sujet de santé"),
        
        # ─── GERMAN (DE) MEDICAL QA ───
        (r"\b(was\s+sind|warum|wie|welche)\b.*\b(symptom|ursache|behandlung|medikament|wirkung)\b", 0.80, "DE: Medizinisches Fragemuster"),
        (r"\b(wofür\s+wird\s+verwendet|wofür\s+ist)\b", 0.85, "DE: Frage zum Medikamentenzweck"),
        (r"\b(bluthochdruck|hoher\s+blutdruck|diabetes|arrhythmie|herzerkrankung)\b", 0.85, "DE: Medizinischer Zustand"),
        (r"\b(medikament|arznei|rezept)\b", 0.75, "DE: Medikamententhema"),
        (r"\b(symptom|zeigt\s+sich\s+mit)\b", 0.80, "DE: Symptomdiskussion"),
        (r"\b(nebenwirkung|unerwünschte\s+reaktion|kontraindikation)\b", 0.85, "DE: Nebenwirkungen"),
        (r"\b(behandlung|therapie|dosierung)\b", 0.75, "DE: Behandlungsrichtlinien"),
        (r"\b(krankheit|störung|syndrom)\b", 0.75, "DE: Gesundheitsthema"),
        
        # ─── PORTUGUESE (PT) MEDICAL QA ───
        (r"\b(quais\s+são|por\s+que|como|qual)\b.*\b(sintoma|causa|tratamento|medicamento|efeito)\b", 0.80, "PT: Padrão de pergunta médica"),
        (r"\b(para\s+que\s+serve|para\s+que\s+é\s+usado)\b", 0.85, "PT: Pergunta sobre uso do medicamento"),
        (r"\b(hipertensão|pressão\s+arterial\s+alta|diabetes|arritmia|doença\s+cardíaca)\b", 0.85, "PT: Condição médica"),
        (r"\b(medicamento|remédio|receita)\b", 0.75, "PT: Tema de medicação"),
        
        # ─── ITALIAN (IT) MEDICAL QA ───
        (r"\b(quali\s+sono|perché|come|quale)\b.*\b(sintomo|causa|trattamento|farmaco|effetto)\b", 0.80, "IT: Schema di domanda medica"),
        (r"\b(a\s+cosa\s+serve|per\s+cosa\s+è\s+usato)\b", 0.85, "IT: Domanda sullo scopo del farmaco"),
        (r"\b(ipertensione|pressione\s+alta|diabete|aritmia|malattia\s+cardiaca)\b", 0.85, "IT: Condizione medica"),
        (r"\b(farmaco|medicinale|ricetta)\b", 0.75, "IT: Argomento di medicazione"),
        
        # ─── CHINESE (ZH) MEDICAL QA ───
        (r"(什么是|为什么|怎么|如何).*(症状|原因|治疗|药物|副作用)", 0.80, "ZH: 医学问题模式"),
        (r"(用于治疗|有什么用|干什么用)", 0.85, "ZH: 药物用途问题"),
        (r"(高血压|糖尿病|心律失常|心脏病)", 0.85, "ZH: 医学条件"),
        (r"(药物|药品|处方)", 0.75, "ZH: 药物主题"),
        
        # ─── JAPANESE (JA) MEDICAL QA ───
        (r"(何が|なぜ|どのように|どうやって).*(症状|原因|治療|薬|副作用)", 0.80, "JA: 医学的質問パターン"),
        (r"(何に使われる|どんな効果)", 0.85, "JA: 薬の目的の質問"),
        (r"(高血圧|糖尿病|不整脈|心臓病)", 0.85, "JA: 医学的状態"),
        (r"(薬|医薬品|処方)", 0.75, "JA: 薬のトピック"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 4.5: DRUG INTERACTION DETECTION (Regex, High Confidence)
    # ═══════════════════════════════════════════════════════════════════════════
    
    DRUG_INTERACTION_PATTERNS = [
        # English explicit interaction queries
        (r"\b(interact|interaction|contraindication|conflict)\b", 0.90, "Explicit interaction query"),
        (r"\b(can\s+i\s+take|safe\s+to\s+take|mix|combine)\b.*?\b(with|and)\b", 0.85, "Combination query"),
        
        # Drug-Drug patterns
        (r"\b(aspirin|warfarin|lisinopril|ibuprofen|metoprolol|sildenafil|nitroglycerin)\b.*?\b(with|and)\b.*?\b(aspirin|warfarin|lisinopril|ibuprofen|metoprolol|sildenafil|nitroglycerin)\b", 0.95, "Specific drug pair"),
        
        # ─── SPANISH (ES) DRUG INTERACTION ───
        (r"\b(interacción|interacciona|contraindicación|conflicto)\b", 0.90, "ES: Consulta de interacción"),
        (r"\b(puedo\s+tomar|es\s+seguro\s+tomar|mezclar|combinar)\b.*?\b(con|y)\b", 0.85, "ES: Consulta de combinación"),
        
        # ─── FRENCH (FR) DRUG INTERACTION ───
        (r"\b(interaction|interagit|contre-indication|conflit)\b", 0.90, "FR: Question d'interaction"),
        (r"\b(puis-je\s+prendre|est-il\s+sûr\s+de\s+prendre|mélanger|combiner)\b.*?\b(avec|et)\b", 0.85, "FR: Question de combinaison"),
        
        # ─── GERMAN (DE) DRUG INTERACTION ───
        (r"\b(wechselwirkung|interaktion|kontraindikation|konflikt)\b", 0.90, "DE: Interaktionsanfrage"),
        (r"\b(kann\s+ich\s+(nehmen|einnehmen)|ist\s+es\s+sicher|mischen|kombinieren)\b.*?\b(mit|und)\b", 0.85, "DE: Kombinationsanfrage"),
        
        # ─── PORTUGUESE (PT) DRUG INTERACTION ───
        (r"\b(interação|contraindicação|conflito)\b", 0.90, "PT: Consulta de interação"),
        (r"\b(posso\s+tomar|é\s+seguro\s+tomar|misturar|combinar)\b.*?\b(com|e)\b", 0.85, "PT: Consulta de combinação"),
        
        # ─── ITALIAN (IT) DRUG INTERACTION ───
        (r"\b(interazione|controindicazione|conflitto)\b", 0.90, "IT: Domanda di interazione"),
        (r"\b(posso\s+prendere|è\s+sicuro\s+prendere|mescolare|combinare)\b.*?\b(con|e)\b", 0.85, "IT: Domanda di combinazione"),
        
        # ─── CHINESE (ZH) DRUG INTERACTION ───
        (r"(相互作用|禁忌|冲突)", 0.90, "ZH: 药物相互作用查询"),
        (r"(可以一起吃|能一起吃|同时服用).*(和|与)", 0.85, "ZH: 药物组合查询"),
        
        # ─── JAPANESE (JA) DRUG INTERACTION ───
        (r"(相互作用|禁忌|飲み合わせ)", 0.90, "JA: 薬物相互作用クエリ"),
        (r"(一緒に飲んで|併用して).*(と|や)", 0.85, "JA: 薬物組み合わせクエリ"),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 5: RESEARCH DETECTION (Regex, Medium Confidence)
    # ═══════════════════════════════════════════════════════════════════════════
    
    RESEARCH_PATTERNS = [
        # Research queries
        (r"\b(research|research\s+on|research\s+about)\b", 0.85, "Research query"),
        (r"\b(find\s+(information|studies|evidence).*?(about|on))\b", 0.80, "Find information"),
        (r"\b(search\s+(for|about))\b", 0.80, "Search query"),
        (r"\b(latest\s+(studies|findings|research|breakthroughs|developments))\b", 0.85, "Latest research"),
        (r"\b(evidence.*?(for|about|on))\b", 0.80, "Evidence request"),
        
        # Clinical/medical research
        (r"\b(clinical\s+trial|clinical\s+study)\b", 0.90, "Clinical trial query"),
        (r"\b(study.*?(about|on)|literature\s+review)\b", 0.80, "Study lookup"),
        (r"\b(systematic\s+review|meta.*analysis|research.*summary)\b", 0.85, "Research summary"),
        
        # Specific conditions research
        (r"\b(what.*new|recent.*breakthrough|latest.*treatment|new.*therapy)\b", 0.80, "New treatment research"),
        (r"\b(cardiology|cardiology.*research|heart.*disease.*research)\b", 0.80, "Cardiology research"),
    ]
    
    def __init__(self, embedding_service=None):
        """
        Initialize router with optional embedding service for edge cases.
        
        Args:
            embedding_service: Optional EmbeddingService for semantic fallback
        """
        self.embedding_service = embedding_service
        
        # P2.2: Route caching for repeated queries
        self._route_cache: Dict[int, Tuple[str, float, str]] = {}
        self._cache_max_size = 500
        
        # Compile regex patterns for performance
        self.compiled_patterns = {
            IntentCategory.EMERGENCY: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.EMERGENCY_PATTERNS
            ],
            IntentCategory.VITALS_QUERY: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.VITALS_PATTERNS
            ],
            IntentCategory.MEDICAL_QA: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.MEDICAL_QA_PATTERNS
            ],
            IntentCategory.RESEARCH: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.RESEARCH_PATTERNS
            ],
            IntentCategory.DRUG_INTERACTION: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.DRUG_INTERACTION_PATTERNS
            ],
            IntentCategory.HEART_RISK: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.HEART_RISK_PATTERNS
            ],
            IntentCategory.TRIAGE: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.TRIAGE_PATTERNS
            ],
            IntentCategory.DIFFERENTIAL_DIAGNOSIS: [
                (re.compile(p, re.IGNORECASE), conf, reason)
                for p, conf, reason in self.DIFFERENTIAL_DIAGNOSIS_PATTERNS
            ],
        }
    
    def route(self, query: str) -> RouteDecision:
        """
        Route query to appropriate handler.
        
        Algorithm:
        1. Emergency detection (highest priority, safety-critical)
        2. Vitals queries (data access)
        3. Medical QA (knowledge)
        4. General (fallback)
        
        Returns:
            RouteDecision with intent, confidence, and target handler
        """
        # P2.2: Normalize query for caching
        query_normalized = query.lower().strip()[:200]
        query_hash = hash(query_normalized)
        
        # P2.2: Check cache first
        if query_hash in self._route_cache:
            cached = self._route_cache[query_hash]
            logger.debug(f"P2.2: Cache hit for query hash {query_hash}")
            return RouteDecision(
                intent=IntentCategory(cached[0]),
                confidence=cached[1],
                target_handler=cached[2],
                reasoning="Cached route",
                requires_embedding=False
            )
        
        query_lower = query.lower()
        logger.debug(f"Routing query: {query}")
        
        # PHASE 1: Regex matching in priority order
        for intent_category in [
            IntentCategory.EMERGENCY,      # 1. Safety-critical first
            IntentCategory.TRIAGE,         # 1.5 Triage assessment
            IntentCategory.HEART_RISK,     # 2. Heart risk assessment
            IntentCategory.DIFFERENTIAL_DIAGNOSIS, # 2.5 Clinical reasoning
            IntentCategory.VITALS_QUERY,   # 3. Data queries
            IntentCategory.DRUG_INTERACTION, # 4. Specific medical checks
            IntentCategory.RESEARCH,       # 4.5 Research queries
            IntentCategory.MEDICAL_QA,     # 5. Knowledge last
        ]:
            result = self._match_patterns(query_lower, intent_category)
            if result and result.confidence >= 0.70:
                logger.info(f"Route decision: {result.intent.value} (confidence: {result.confidence})")
                
                # P2.2: Cache high-confidence results
                if result.confidence >= 0.70:
                    # Limit cache size
                    if len(self._route_cache) >= self._cache_max_size:
                        # Remove oldest (first) entry
                        oldest_key = next(iter(self._route_cache))
                        del self._route_cache[oldest_key]
                    
                    self._route_cache[query_hash] = (
                        result.intent.value,
                        result.confidence,
                        result.target_handler
                    )
                
                return result
        
        # PHASE 2: If no high-confidence regex match, fallback to general
        return RouteDecision(
            intent=IntentCategory.GENERAL,
            confidence=0.0,
            target_handler="general_chat",
            reasoning="No specific intent detected, routing to general chat",
            requires_embedding=False
        )
    
    def _match_patterns(
        self,
        query: str,
        intent: IntentCategory
    ) -> Optional[RouteDecision]:
        """
        Match query against patterns for given intent.
        
        Returns first match with highest confidence.
        """
        patterns = self.compiled_patterns.get(intent, [])
        
        best_match = None
        
        for pattern, confidence, reason in patterns:
            if pattern.search(query):
                if best_match is None or confidence > best_match[1]:
                    best_match = (pattern.pattern, confidence, reason)
        
        if best_match:
            pattern_str, confidence, reason = best_match
            return RouteDecision(
                intent=intent,
                confidence=confidence,
                target_handler=self._get_handler_name(intent),
                reasoning=reason,
                requires_embedding=False,
                matched_pattern=pattern_str
            )
        
        return None
    
    @staticmethod
    def _get_handler_name(intent: IntentCategory) -> str:
        """Map intent to handler/tool name."""
        handlers = {
            IntentCategory.EMERGENCY: "emergency_handler",
            IntentCategory.HEART_RISK: "heart_analyst",
            IntentCategory.VITALS_QUERY: "sql_tool",
            IntentCategory.DRUG_INTERACTION: "drug_checker_tool",
            IntentCategory.DIFFERENTIAL_DIAGNOSIS: "clinical_reasoning",
            IntentCategory.TRIAGE: "clinical_reasoning",
            IntentCategory.MEDICAL_QA: "rag_tool",
            IntentCategory.RESEARCH: "researcher",
            IntentCategory.GENERAL: "general_chat",
        }
        return handlers.get(intent, "general_chat")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: Keep existing SemanticRouter name
# ═══════════════════════════════════════════════════════════════════════════════

SemanticRouter = SemanticRouterV2  # Alias for existing code


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick test of semantic router.
    
    Run with: python -m tools.semantic_router_v2
    """
    router = SemanticRouterV2()
    
    # Test cases
    test_queries = [
        # Emergency
        ("I can't breathe", IntentCategory.EMERGENCY, 0.95),
        ("chest pain emergency", IntentCategory.EMERGENCY, 0.95),
        ("call 911", IntentCategory.EMERGENCY, 0.90),
        
        # Vitals
        ("What was my heart rate last week?", IntentCategory.VITALS_QUERY, 0.85),
        ("Show my blood pressure readings", IntentCategory.VITALS_QUERY, 0.85),
        ("What's my current BP?", IntentCategory.VITALS_QUERY, 0.90),
        
        # Medical QA
        ("What are the side effects of Lisinopril?", IntentCategory.MEDICAL_QA, 0.85),
        ("How do I treat hypertension?", IntentCategory.MEDICAL_QA, 0.80),
        
        # General
        ("Hi there", IntentCategory.GENERAL, 0.0),
        ("Thanks!", IntentCategory.GENERAL, 0.0),
    ]
    
    print("=" * 80)
    print("SEMANTIC ROUTER V2 - TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    for query, expected_intent, min_confidence in test_queries:
        decision = router.route(query)
        
        is_correct = decision.intent == expected_intent and decision.confidence >= min_confidence
        status = "✅ PASS" if is_correct else "❌ FAIL"
        
        print(f"\n{status}")
        print(f"  Query: '{query}'")
        print(f"  Expected: {expected_intent.value} (confidence >= {min_confidence})")
        print(f"  Got: {decision.intent.value} (confidence: {decision.confidence})")
        print(f"  Reasoning: {decision.reasoning}")
        
        if is_correct:
            passed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_queries)} tests passed")
    print("=" * 80)
