import { HeartDiseasePredictionResponse, HeartDiseasePredictionRequest } from './api.types';
import { jsPDF } from 'jspdf';

// ============================================================================
// Types
// ============================================================================

export interface SavedAssessment {
    id: string;
    timestamp: string;
    input: HeartDiseasePredictionRequest;
    result: HeartDiseasePredictionResponse;
}

const STORAGE_KEY = 'heart_saved_assessments';

// ============================================================================
// CRUD Operations
// ============================================================================

export function saveAssessment(input: HeartDiseasePredictionRequest, result: HeartDiseasePredictionResponse): SavedAssessment {
    const saved: SavedAssessment = {
        id: `assessment_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        timestamp: new Date().toISOString(),
        input,
        result,
    };

    const existing = getSavedAssessments();
    existing.unshift(saved); // newest first
    // Keep max 50 assessments
    if (existing.length > 50) existing.length = 50;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
    return saved;
}

export function getSavedAssessments(): SavedAssessment[] {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        return JSON.parse(raw) as SavedAssessment[];
    } catch {
        return [];
    }
}

export function getAssessmentById(id: string): SavedAssessment | null {
    return getSavedAssessments().find(a => a.id === id) || null;
}

export function deleteAssessment(id: string): void {
    const list = getSavedAssessments().filter(a => a.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

export function clearAllAssessments(): void {
    localStorage.removeItem(STORAGE_KEY);
}

// ============================================================================
// Sharing
// ============================================================================

function buildShareText(saved: SavedAssessment): string {
    const r = saved.result;
    const prob = (r.probability * 100).toFixed(1);
    const conf = r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : 'N/A';
    const date = new Date(saved.timestamp).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });

    let text = `🫀 Heart Risk Assessment Report\n`;
    text += `📅 ${date}\n\n`;
    text += `━━━ Results ━━━\n`;
    text += `Risk Level: ${r.risk_level || (r.prediction === 1 ? 'High' : 'Low')}\n`;
    text += `Probability: ${prob}%\n`;
    text += `Confidence: ${conf}\n`;
    text += `Prediction: ${r.prediction === 1 ? 'Heart Disease Likely' : 'Heart Disease Unlikely'}\n\n`;

    const chestPainMap: Record<number, string> = { 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic' };
    const ecgMap: Record<number, string> = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LVH' };
    const slopeMap: Record<number, string> = { 1: 'Upsloping', 2: 'Flat', 3: 'Downsloping' };

    text += `━━━ Patient Data ━━━\n`;
    text += `Age: ${saved.input.age}\n`;
    text += `Sex: ${saved.input.sex === 1 ? 'Male' : 'Female'}\n`;
    text += `Resting BP: ${saved.input.resting_bp_s} mm Hg\n`;
    text += `Cholesterol: ${saved.input.cholesterol} mg/dl\n`;
    text += `Max Heart Rate: ${saved.input.max_heart_rate} bpm\n`;
    text += `Fasting Blood Sugar: ${saved.input.fasting_blood_sugar === 1 ? '> 120 mg/dl' : '<= 120 mg/dl'}\n`;
    text += `Chest Pain Type: ${chestPainMap[saved.input.chest_pain_type] || saved.input.chest_pain_type}\n`;
    text += `Resting ECG: ${ecgMap[saved.input.resting_ecg] || saved.input.resting_ecg}\n`;
    text += `Exercise Angina: ${saved.input.exercise_angina === 1 ? 'Yes' : 'No'}\n`;
    text += `Oldpeak: ${saved.input.oldpeak}\n`;
    text += `ST Slope: ${slopeMap[saved.input.st_slope] || saved.input.st_slope}\n\n`;

    if (r.message) {
        text += `━━━ Summary ━━━\n${r.message}\n\n`;
    }

    if (r.clinical_interpretation) {
        text += `━━━ Clinical Interpretation ━━━\n${r.clinical_interpretation}\n\n`;
    }

    text += `⚠️ This is an AI-generated assessment for informational purposes only. Consult a cardiologist for medical advice.`;
    return text;
}

export function shareViaWhatsApp(saved: SavedAssessment): void {
    const text = encodeURIComponent(buildShareText(saved));
    window.open(`https://wa.me/?text=${text}`, '_blank');
}

export function shareViaLink(saved: SavedAssessment): Promise<void> {
    const text = buildShareText(saved);

    if (navigator.share) {
        return navigator.share({
            title: 'Heart Risk Assessment Report',
            text,
        }).catch(() => {
            // Fallback: copy to clipboard
            return copyToClipboard(text);
        });
    }

    return copyToClipboard(text);
}

async function copyToClipboard(text: string): Promise<void> {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
}

// ============================================================================
// PDF Download
// ============================================================================

export function downloadAsPDF(saved: SavedAssessment): void {
    const r = saved.result;
    const prob = (r.probability * 100).toFixed(1);
    const conf = r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : 'N/A';
    const date = new Date(saved.timestamp).toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });

    const riskColor = r.prediction === 1 ? '#dc2626' : '#16a34a';
    const riskBg = r.prediction === 1 ? '#fef2f2' : '#f0fdf4';
    const riskLabel = r.risk_level || (r.prediction === 1 ? 'High' : 'Low');

    const chestPainMap: Record<number, string> = { 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic' };
    const ecgMap: Record<number, string> = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LVH' };
    const slopeMap: Record<number, string> = { 1: 'Upsloping', 2: 'Flat', 3: 'Downsloping' };

    // Format clinical interpretation
    let interpretationHtml = '';
    if (r.clinical_interpretation) {
        interpretationHtml = r.clinical_interpretation
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\* /g, '\n<br>• ')
            .replace(/\n/g, '<br>');
    }

    // Build test results table
    let testResultsHtml = '';
    if (r.test_results && r.test_results.length > 0) {
        testResultsHtml = `
            <h3 style="color:#1e293b;margin:24px 0 12px;font-size:16px;">📋 Detailed Test Results</h3>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <thead>
                    <tr style="background:#f1f5f9;">
                        <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #e2e8f0;">Test</th>
                        <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #e2e8f0;">Value</th>
                        <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #e2e8f0;">Status</th>
                        <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #e2e8f0;">Risk</th>
                    </tr>
                </thead>
                <tbody>
                    ${r.test_results.map(t => {
            const statusColor = t.status === 'Normal' ? '#16a34a' : t.status === 'Critical' ? '#dc2626' : t.status === 'Borderline' ? '#d97706' : '#ea580c';
            return `<tr style="border-bottom:1px solid #f1f5f9;">
                            <td style="padding:8px 12px;font-weight:600;">${t.test_name}</td>
                            <td style="padding:8px 12px;">${t.value}</td>
                            <td style="padding:8px 12px;color:${statusColor};font-weight:600;">${t.status}</td>
                            <td style="padding:8px 12px;">${t.risk_contribution}</td>
                        </tr>`;
        }).join('')}
                </tbody>
            </table>`;
    }

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Risk Assessment Report</title>
    <style>
        @media print {
            body { margin: 0; padding: 20px; }
            .no-print { display: none !important; }
        }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1e293b; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.6; }
        .header { text-align: center; padding: 30px; background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; border-radius: 16px; margin-bottom: 30px; }
        .header h1 { margin: 0 0 4px; font-size: 24px; }
        .header p { margin: 0; opacity: 0.9; font-size: 14px; }
        .risk-card { background: ${riskBg}; border: 2px solid ${riskColor}20; border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px; }
        .risk-label { display: inline-block; background: ${riskColor}; color: white; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
        .metrics { display: flex; gap: 16px; justify-content: center; margin-top: 16px; }
        .metric { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 24px; text-align: center; }
        .metric-value { font-size: 28px; font-weight: 800; color: ${riskColor}; }
        .metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 700; }
        .section { margin-bottom: 24px; }
        .section h3 { color: #1e293b; font-size: 16px; margin: 0 0 12px; }
        .data-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .data-item { background: #f8fafc; padding: 12px 16px; border-radius: 10px; display: flex; justify-content: space-between; }
        .data-label { color: #64748b; font-size: 13px; }
        .data-value { font-weight: 700; color: #1e293b; font-size: 13px; }
        .interpretation { background: #f8fafc; border-radius: 12px; padding: 20px; font-size: 13px; line-height: 1.8; border-left: 4px solid #6366f1; }
        .footer { text-align: center; padding: 20px; color: #94a3b8; font-size: 11px; border-top: 1px solid #e2e8f0; margin-top: 30px; }
        .print-btn { display: block; margin: 20px auto; padding: 12px 32px; background: #3b82f6; color: white; border: none; border-radius: 10px; font-weight: 700; font-size: 14px; cursor: pointer; }
        .print-btn:hover { background: #2563eb; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🫀 Heart Risk Assessment Report</h1>
        <p>HeartGuard AI — Cardio Clinical Prediction</p>
        <p style="margin-top:8px;font-size:12px;opacity:0.8;">${date}</p>
    </div>

    <div class="risk-card">
        <span class="risk-label">${riskLabel} Risk</span>
        <p style="color:${riskColor};font-size:14px;margin:12px 0 0;font-weight:600;">${r.message || ''}</p>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Probability</div>
                <div class="metric-value">${prob}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color:#3b82f6;">${conf}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>🧑‍⚕️ Patient Data</h3>
        <div class="data-grid">
            <div class="data-item"><span class="data-label">Age</span><span class="data-value">${saved.input.age} years</span></div>
            <div class="data-item"><span class="data-label">Sex</span><span class="data-value">${saved.input.sex === 1 ? 'Male' : 'Female'}</span></div>
            <div class="data-item"><span class="data-label">Resting BP</span><span class="data-value">${saved.input.resting_bp_s} mm Hg</span></div>
            <div class="data-item"><span class="data-label">Cholesterol</span><span class="data-value">${saved.input.cholesterol} mg/dl</span></div>
            <div class="data-item"><span class="data-label">Max Heart Rate</span><span class="data-value">${saved.input.max_heart_rate} bpm</span></div>
            <div class="data-item"><span class="data-label">Fasting Blood Sugar</span><span class="data-value">${saved.input.fasting_blood_sugar === 1 ? '> 120' : '≤ 120'} mg/dl</span></div>
            <div class="data-item"><span class="data-label">Chest Pain</span><span class="data-value">${chestPainMap[saved.input.chest_pain_type] || saved.input.chest_pain_type}</span></div>
            <div class="data-item"><span class="data-label">Resting ECG</span><span class="data-value">${ecgMap[saved.input.resting_ecg] || saved.input.resting_ecg}</span></div>
            <div class="data-item"><span class="data-label">Exercise Angina</span><span class="data-value">${saved.input.exercise_angina === 1 ? 'Yes' : 'No'}</span></div>
            <div class="data-item"><span class="data-label">Oldpeak</span><span class="data-value">${saved.input.oldpeak}</span></div>
            <div class="data-item"><span class="data-label">ST Slope</span><span class="data-value">${slopeMap[saved.input.st_slope] || saved.input.st_slope}</span></div>
        </div>
    </div>

    ${testResultsHtml}

    ${interpretationHtml ? `
    <div class="section">
        <h3>🧠 AI Clinical Interpretation</h3>
        <div class="interpretation">${interpretationHtml}</div>
    </div>` : ''}

    <div class="footer">
        <p>⚠️ This AI-generated report is for informational purposes only.<br>
        It does not constitute medical advice, diagnosis, or treatment.<br>
        Please consult a qualified cardiologist for clinical decisions.</p>
        <p style="margin-top:8px;">Generated by HeartGuard AI • ${date}</p>
    </div>

    <button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
</body>
</html>`;

    // Open in new window and trigger print
    const printWindow = window.open('', '_blank');
    if (printWindow) {
        printWindow.document.write(html);
        printWindow.document.close();
        // Slight delay to let styles load
        setTimeout(() => printWindow.print(), 500);
    }
}

// ============================================================================
// Share as PDF (real PDF blob via jsPDF + Web Share API)
// ============================================================================

function buildPDFReport(saved: SavedAssessment): jsPDF {
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const r = saved.result;
    const prob = (r.probability * 100).toFixed(1);
    const conf = r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : 'N/A';
    const date = new Date(saved.timestamp).toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    const riskLabel = r.risk_level || (r.prediction === 1 ? 'High' : 'Low');
    const isHighRisk = r.prediction === 1;

    const chestPainMap: Record<number, string> = { 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic' };
    const ecgMap: Record<number, string> = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LVH' };
    const slopeMap: Record<number, string> = { 1: 'Upsloping', 2: 'Flat', 3: 'Downsloping' };

    const pageWidth = 210;
    const margin = 20;
    const contentWidth = pageWidth - 2 * margin;
    let y = 20;

    // Helper to check page overflow and add new page
    const checkPageBreak = (height: number) => {
        if (y + height > 275) {
            doc.addPage();
            y = 20;
        }
    };

    // Header
    doc.setFillColor(59, 130, 246);
    doc.rect(0, 0, pageWidth, 40, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('Heart Risk Assessment Report', pageWidth / 2, 18, { align: 'center' });
    doc.setFontSize(11);
    doc.setFont('helvetica', 'normal');
    doc.text('HeartGuard AI — Cardio Clinical Prediction', pageWidth / 2, 26, { align: 'center' });
    doc.setFontSize(9);
    doc.text(date, pageWidth / 2, 34, { align: 'center' });

    y = 50;

    // Risk Card
    const riskBgColor: [number, number, number] = isHighRisk ? [254, 242, 242] : [240, 253, 244];
    const riskTextColor: [number, number, number] = isHighRisk ? [220, 38, 38] : [22, 163, 74];
    doc.setFillColor(...riskBgColor);
    doc.roundedRect(margin, y, contentWidth, 30, 4, 4, 'F');
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...riskTextColor);
    doc.text(`${riskLabel} Risk`, pageWidth / 2, y + 12, { align: 'center' });
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Probability: ${prob}%  |  Confidence: ${conf}`, pageWidth / 2, y + 22, { align: 'center' });
    if (r.message) {
        doc.setFontSize(9);
        doc.text(r.message.substring(0, 90), pageWidth / 2, y + 28, { align: 'center' });
    }

    y += 38;

    // Patient Data Section
    checkPageBreak(60);
    doc.setFontSize(13);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(30, 41, 59);
    doc.text('Patient Data', margin, y);
    y += 8;

    const patientData: [string, string][] = [
        ['Age', `${saved.input.age} years`],
        ['Sex', saved.input.sex === 1 ? 'Male' : 'Female'],
        ['Resting BP', `${saved.input.resting_bp_s} mm Hg`],
        ['Cholesterol', `${saved.input.cholesterol} mg/dl`],
        ['Max Heart Rate', `${saved.input.max_heart_rate} bpm`],
        ['Fasting Blood Sugar', saved.input.fasting_blood_sugar === 1 ? '> 120 mg/dl' : '<= 120 mg/dl'],
        ['Chest Pain Type', chestPainMap[saved.input.chest_pain_type] || String(saved.input.chest_pain_type)],
        ['Resting ECG', ecgMap[saved.input.resting_ecg] || String(saved.input.resting_ecg)],
        ['Exercise Angina', saved.input.exercise_angina === 1 ? 'Yes' : 'No'],
        ['Oldpeak', String(saved.input.oldpeak)],
        ['ST Slope', slopeMap[saved.input.st_slope] || String(saved.input.st_slope)],
    ];

    doc.setFontSize(10);
    for (const [label, value] of patientData) {
        checkPageBreak(8);
        // alternating row bg
        if (patientData.indexOf([label, value] as any) % 2 === 0) {
            doc.setFillColor(248, 250, 252);
            doc.rect(margin, y - 4, contentWidth, 7, 'F');
        }
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(100, 116, 139);
        doc.text(label, margin + 2, y);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(30, 41, 59);
        doc.text(value, margin + contentWidth - 2, y, { align: 'right' });
        y += 7;
    }

    y += 6;

    // Test Results Table
    if (r.test_results && r.test_results.length > 0) {
        checkPageBreak(20);
        doc.setFontSize(13);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(30, 41, 59);
        doc.text('Detailed Test Results', margin, y);
        y += 8;

        // Table header
        doc.setFillColor(241, 245, 249);
        doc.rect(margin, y - 4, contentWidth, 7, 'F');
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(71, 85, 105);
        doc.text('Test', margin + 2, y);
        doc.text('Value', margin + 60, y);
        doc.text('Status', margin + 100, y);
        doc.text('Risk', margin + 130, y);
        y += 7;

        doc.setFont('helvetica', 'normal');
        for (const t of r.test_results) {
            checkPageBreak(7);
            doc.setTextColor(30, 41, 59);
            doc.text(t.test_name.substring(0, 25), margin + 2, y);
            doc.text(String(t.value).substring(0, 20), margin + 60, y);
            const statusColor: [number, number, number] = t.status === 'Normal' ? [22, 163, 74] : t.status === 'Critical' ? [220, 38, 38] : [217, 119, 6];
            doc.setTextColor(...statusColor);
            doc.text(t.status, margin + 100, y);
            doc.setTextColor(30, 41, 59);
            doc.text(t.risk_contribution.substring(0, 20), margin + 130, y);
            y += 7;
        }
        y += 6;
    }

    // Clinical Interpretation
    if (r.clinical_interpretation) {
        checkPageBreak(20);
        doc.setFontSize(13);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(30, 41, 59);
        doc.text('AI Clinical Interpretation', margin, y);
        y += 8;

        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(51, 65, 85);
        // Clean markdown formatting
        const cleanText = r.clinical_interpretation
            .replace(/\*\*/g, '')
            .replace(/\*/g, '•');
        const lines = doc.splitTextToSize(cleanText, contentWidth - 4);
        for (const line of lines) {
            checkPageBreak(5);
            doc.text(line, margin + 2, y);
            y += 5;
        }
        y += 6;
    }

    // Footer
    checkPageBreak(20);
    doc.setDrawColor(226, 232, 240);
    doc.line(margin, y, margin + contentWidth, y);
    y += 8;
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);
    doc.setFont('helvetica', 'normal');
    const footerLines = [
        'This AI-generated report is for informational purposes only.',
        'It does not constitute medical advice, diagnosis, or treatment.',
        'Please consult a qualified cardiologist for clinical decisions.',
        `Generated by HeartGuard AI • ${date}`,
    ];
    for (const line of footerLines) {
        doc.text(line, pageWidth / 2, y, { align: 'center' });
        y += 4;
    }

    return doc;
}

/**
 * Share the assessment report as a PDF file using Web Share API.
 * Falls back to download if sharing is not supported.
 */
export async function shareAsPDF(saved: SavedAssessment): Promise<void> {
    const doc = buildPDFReport(saved);
    const riskLabel = saved.result.risk_level || (saved.result.prediction === 1 ? 'High' : 'Low');
    const filename = `HeartGuard_Report_${riskLabel}_${new Date(saved.timestamp).toISOString().slice(0, 10)}.pdf`;

    // Generate blob
    const pdfBlob = doc.output('blob');
    const pdfFile = new File([pdfBlob], filename, { type: 'application/pdf' });

    // Try Web Share API with file support
    if (navigator.canShare && navigator.canShare({ files: [pdfFile] })) {
        try {
            await navigator.share({
                title: 'Heart Risk Assessment Report',
                text: `HeartGuard AI Report — ${riskLabel} Risk`,
                files: [pdfFile],
            });
            return;
        } catch (err: any) {
            // User cancelled or share failed — fall back to download
            if (err?.name === 'AbortError') return;
            console.warn('[SharePDF] Web Share failed, falling back to download:', err);
        }
    }

    // Fallback: trigger browser download
    const url = URL.createObjectURL(pdfBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
