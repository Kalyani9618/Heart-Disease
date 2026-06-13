"""
Workflow Automation Engine - Medical Data Routing and Report Generation
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)



class UrgencyLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NORMAL = "normal"


class FindingCategory(Enum):
    CRITICAL_ANOMALY = "critical_anomaly"
    ABNORMAL = "abnormal"
    BORDERLINE = "borderline"
    NORMAL = "normal"
    INCONCLUSIVE = "inconclusive"


@dataclass
class MedicalFinding:
    finding_id: str
    modality: str
    description: str
    category: FindingCategory
    confidence: float
    urgency: UrgencyLevel
    location: Optional[str] = None
    measurements: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    raw_data: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RoutingDecision:
    finding_id: str
    destination: str
    urgency: UrgencyLevel
    action_required: str
    auto_escalate: bool
    notification_channels: List[str]
    report_drafted: Optional[str] = None


class WorkflowRouter:
    """Routes medical findings to appropriate handlers based on urgency and type."""

    ROUTING_RULES = {
        FindingCategory.CRITICAL_ANOMALY: {
            "destination": "on_call_specialist",
            "auto_escalate": True,
            "notification_channels": ["pager", "phone", "sms"],
            "sla_minutes": 5,
        },
        FindingCategory.ABNORMAL: {
            "destination": "specialist_queue",
            "auto_escalate": False,
            "notification_channels": ["email", "dashboard"],
            "sla_minutes": 60,
        },
        FindingCategory.BORDERLINE: {
            "destination": "senior_review",
            "auto_escalate": False,
            "notification_channels": ["email"],
            "sla_minutes": 240,
        },
        FindingCategory.NORMAL: {
            "destination": "auto_report",
            "auto_escalate": False,
            "notification_channels": [],
            "sla_minutes": 480,
        },
        FindingCategory.INCONCLUSIVE: {
            "destination": "additional_imaging",
            "auto_escalate": False,
            "notification_channels": ["email"],
            "sla_minutes": 120,
        },
    }

    SPECIALIST_MAP = {
        "chest_xray": "radiologist",
        "ct_scan": "radiologist",
        "mri": "radiologist",
        "ecg": "cardiologist",
        "echo": "cardiologist",
        "pathology": "pathologist",
        "dermatology": "dermatologist",
        "fundus": "ophthalmologist",
    }

    def __init__(self, llm_gateway=None, notification_service=None):
        self.llm_gateway = llm_gateway
        self.notification_service = notification_service
        self._callbacks: Dict[str, List[Callable]] = {}

    async def route(self, finding: MedicalFinding) -> RoutingDecision:
        """Route a medical finding to the appropriate handler."""
        rule = self.ROUTING_RULES.get(finding.category, self.ROUTING_RULES[FindingCategory.NORMAL])
        
        specialist = self.SPECIALIST_MAP.get(finding.modality, "general_physician")
        destination = f"{specialist}_{rule['destination']}" if rule["destination"] != "auto_report" else "auto_report"
        
        action = self._determine_action(finding, rule)
        
        report = None
        if finding.category == FindingCategory.NORMAL and self.llm_gateway:
            report = await self._draft_normal_report(finding)
        
        decision = RoutingDecision(
            finding_id=finding.finding_id,
            destination=destination,
            urgency=finding.urgency,
            action_required=action,
            auto_escalate=rule["auto_escalate"],
            notification_channels=rule["notification_channels"],
            report_drafted=report,
        )
        
        await self._execute_routing(decision, finding)
        logger.info(f"Routed finding {finding.finding_id} to {destination}")
        
        return decision

    def _determine_action(self, finding: MedicalFinding, rule: Dict) -> str:
        if finding.category == FindingCategory.CRITICAL_ANOMALY:
            return f"IMMEDIATE REVIEW REQUIRED: {finding.description}"
        elif finding.category == FindingCategory.ABNORMAL:
            return f"Review within {rule['sla_minutes']} minutes: {finding.description}"
        elif finding.category == FindingCategory.NORMAL:
            return f"Verify auto-drafted report for: {finding.modality}"
        elif finding.category == FindingCategory.INCONCLUSIVE:
            return f"Consider additional imaging: {finding.recommendations[0] if finding.recommendations else 'Further evaluation needed'}"
        return "Standard review"

    async def _draft_normal_report(self, finding: MedicalFinding) -> str:
        """Auto-draft a normal findings report using MedGemma."""
        return f"""
PRELIMINARY REPORT - PENDING PHYSICIAN REVIEW

EXAMINATION: {finding.modality.replace('_', ' ').title()}
DATE: {finding.timestamp[:10]}

FINDINGS:
{finding.description}

IMPRESSION:
No acute abnormality identified. Findings are within normal limits.
Confidence Score: {finding.confidence * 100:.1f}%

---
⚠️ This is an AI-generated preliminary report.
Final interpretation by a licensed physician is required.
"""

    async def _execute_routing(self, decision: RoutingDecision, finding: MedicalFinding) -> None:
        if not self.notification_service:
            return
        
        for channel in decision.notification_channels:
            try:
                await self.notification_service.send(
                    channel=channel,
                    message=f"[{decision.urgency.value.upper()}] {decision.action_required}",
                    metadata={"finding_id": finding.finding_id, "destination": decision.destination}
                )
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")


class AnomalyTriggeredRouter:
    """Wrapper that integrates anomaly detection with workflow routing."""

    def __init__(self, workflow_router: WorkflowRouter, llm_gateway=None):
        self.router = workflow_router
        self.llm_gateway = llm_gateway

    async def analyze_and_route(
        self,
        modality: str,
        analysis_result: Dict[str, Any],
        raw_data: Optional[Dict] = None,
    ) -> RoutingDecision:
        category = self._categorize_finding(analysis_result)
        urgency = self._determine_urgency(analysis_result, category)
        
        finding = MedicalFinding(
            finding_id=f"finding-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            modality=modality,
            description=analysis_result.get("description", "No description provided"),
            category=category,
            confidence=analysis_result.get("confidence", 0.0),
            urgency=urgency,
            location=analysis_result.get("location"),
            measurements=analysis_result.get("measurements", {}),
            recommendations=analysis_result.get("recommendations", []),
            raw_data=raw_data,
        )
        
        return await self.router.route(finding)

    def _categorize_finding(self, analysis: Dict) -> FindingCategory:
        confidence = analysis.get("confidence", 0)
        is_abnormal = analysis.get("is_abnormal", False)
        is_critical = analysis.get("is_critical", False)
        
        if is_critical and confidence > 0.8:
            return FindingCategory.CRITICAL_ANOMALY
        elif is_abnormal and confidence > 0.7:
            return FindingCategory.ABNORMAL
        elif is_abnormal and confidence > 0.5:
            return FindingCategory.BORDERLINE
        elif confidence < 0.5:
            return FindingCategory.INCONCLUSIVE
        else:
            return FindingCategory.NORMAL

    def _determine_urgency(self, analysis: Dict, category: FindingCategory) -> UrgencyLevel:
        urgency_map = {
            FindingCategory.CRITICAL_ANOMALY: UrgencyLevel.CRITICAL,
            FindingCategory.ABNORMAL: UrgencyLevel.HIGH,
            FindingCategory.BORDERLINE: UrgencyLevel.MODERATE,
            FindingCategory.INCONCLUSIVE: UrgencyLevel.MODERATE,
            FindingCategory.NORMAL: UrgencyLevel.NORMAL,
        }
        
        keywords = analysis.get("description", "").lower()
        critical_terms = ["pneumothorax", "aortic dissection", "stemi", "stroke", "hemorrhage"]
        
        if any(term in keywords for term in critical_terms):
            return UrgencyLevel.CRITICAL
        
        return urgency_map.get(category, UrgencyLevel.NORMAL)
