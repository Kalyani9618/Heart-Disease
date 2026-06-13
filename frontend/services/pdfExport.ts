/**
 * PDF Export Service
 *
 * Generates PDF reports for health data and chat history
 * Uses jsPDF for PDF generation
 */

import { jsPDF } from 'jspdf';
import { HealthAssessment, Message, Medication, Appointment } from '../types';

// ============================================================================
// Types
// ============================================================================

interface HealthReport {
  user?: {
    name: string;
    email?: string;
  };
  assessment?: HealthAssessment;
  medications?: Medication[];
  appointments?: Appointment[];
  biometricHistory?: BiometricEntry[];
}

interface BiometricEntry {
  date: string;
  systolic?: number;
  diastolic?: number;
  heartRate?: number;
  weight?: number;
}

interface ChatExport {
  sessionId?: string;
  messages: Message[];
  exportDate: Date;
}

// ============================================================================
// Color Palette (matches app theme)
// ============================================================================

const COLORS = {
  primary: '#D32F2F',      // Red accent
  secondary: '#192633',    // Dark background
  text: '#1F2937',         // Dark text
  textLight: '#6B7280',    // Light text
  border: '#E5E7EB',       // Light border
  success: '#10B981',      // Green
  warning: '#F59E0B',      // Amber
  danger: '#EF4444',       // Red
  white: '#FFFFFF',
};

// ============================================================================
// Helper Functions
// ============================================================================

function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

function formatDateTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [0, 0, 0];
}

// ============================================================================
// PDF Generation Class
// ============================================================================

class PDFExportService {
  private doc: jsPDF | null = null;
  private currentY = 20;
  private pageMargin = 20;
  private pageWidth = 210; // A4 width in mm
  private contentWidth = 170; // pageWidth - 2 * margin

  /**
   * Initialize a new PDF document
   */
  private initDocument(title: string): jsPDF {
    this.doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    this.currentY = 20;
    this.addHeader(title);
    return this.doc;
  }

  /**
   * Add page header
   */
  private addHeader(title: string): void {
    if (!this.doc) return;

    // Logo area (heart icon representation)
    const [r, g, b] = hexToRgb(COLORS.primary);
    this.doc.setFillColor(r, g, b);
    this.doc.circle(this.pageMargin + 8, 15, 8, 'F');

    // Title
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(20);
    this.doc.setTextColor(...hexToRgb(COLORS.text));
    this.doc.text('Cardio AI', this.pageMargin + 22, 18);

    // Subtitle
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(12);
    this.doc.setTextColor(...hexToRgb(COLORS.textLight));
    this.doc.text(title, this.pageMargin + 22, 25);

    // Date
    this.doc.setFontSize(10);
    this.doc.text(`Generated: ${formatDateTime(new Date())}`, this.pageWidth - this.pageMargin, 18, { align: 'right' });

    // Divider line
    this.doc.setDrawColor(...hexToRgb(COLORS.border));
    this.doc.setLineWidth(0.5);
    this.doc.line(this.pageMargin, 32, this.pageWidth - this.pageMargin, 32);

    this.currentY = 45;
  }

  /**
   * Add a section title
   */
  private addSectionTitle(title: string): void {
    if (!this.doc) return;

    this.checkPageBreak(15);

    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(14);
    this.doc.setTextColor(...hexToRgb(COLORS.primary));
    this.doc.text(title, this.pageMargin, this.currentY);
    this.currentY += 8;
  }

  /**
   * Add regular text
   */
  private addText(text: string, options?: { bold?: boolean; color?: string; size?: number }): void {
    if (!this.doc) return;

    const { bold = false, color = COLORS.text, size = 11 } = options || {};

    this.checkPageBreak(8);

    this.doc.setFont('helvetica', bold ? 'bold' : 'normal');
    this.doc.setFontSize(size);
    this.doc.setTextColor(...hexToRgb(color));

    // Wrap text if needed
    const lines = this.doc.splitTextToSize(text, this.contentWidth);
    this.doc.text(lines, this.pageMargin, this.currentY);
    this.currentY += lines.length * (size * 0.4) + 2;
  }

  /**
   * Add a key-value pair
   */
  private addKeyValue(key: string, value: string, options?: { valueColor?: string }): void {
    if (!this.doc) return;

    this.checkPageBreak(8);

    const { valueColor = COLORS.text } = options || {};

    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(10);
    this.doc.setTextColor(...hexToRgb(COLORS.textLight));
    this.doc.text(`${key}:`, this.pageMargin, this.currentY);

    this.doc.setFont('helvetica', 'normal');
    this.doc.setTextColor(...hexToRgb(valueColor));
    this.doc.text(value, this.pageMargin + 40, this.currentY);

    this.currentY += 6;
  }

  /**
   * Add a card/box with content
   */
  private addCard(title: string, content: string, color: string = COLORS.secondary): void {
    if (!this.doc) return;

    this.checkPageBreak(25);

    // Card background
    this.doc.setFillColor(...hexToRgb('#F3F4F6'));
    this.doc.roundedRect(this.pageMargin, this.currentY - 2, this.contentWidth, 20, 3, 3, 'F');

    // Card title
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(10);
    this.doc.setTextColor(...hexToRgb(color));
    this.doc.text(title, this.pageMargin + 4, this.currentY + 5);

    // Card content
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(11);
    this.doc.setTextColor(...hexToRgb(COLORS.text));
    this.doc.text(content, this.pageMargin + 4, this.currentY + 13);

    this.currentY += 25;
  }

  /**
   * Add spacing
   */
  private addSpace(height: number = 5): void {
    this.currentY += height;
  }

  /**
   * Check if we need a page break
   */
  private checkPageBreak(neededSpace: number): void {
    if (!this.doc) return;

    if (this.currentY + neededSpace > 280) { // A4 height is 297mm, leave margin
      this.doc.addPage();
      this.currentY = 20;
    }
  }

  /**
   * Add page footer
   */
  private addFooter(): void {
    if (!this.doc) return;

    const pageCount = this.doc.getNumberOfPages();

    for (let i = 1; i <= pageCount; i++) {
      this.doc.setPage(i);
      this.doc.setFont('helvetica', 'normal');
      this.doc.setFontSize(8);
      this.doc.setTextColor(...hexToRgb(COLORS.textLight));
      this.doc.text(
        `Page ${i} of ${pageCount}`,
        this.pageWidth / 2,
        290,
        { align: 'center' }
      );
      this.doc.text(
        'Confidential Health Information',
        this.pageMargin,
        290
      );
    }
  }

  // ============================================================================
  // Public Export Methods
  // ============================================================================

  /**
   * Export health report as PDF
   */
  exportHealthReport(data: HealthReport): void {
    this.initDocument('Health Report');

    // User info section
    if (data.user) {
      this.addSectionTitle('Patient Information');
      this.addKeyValue('Name', data.user.name);
      if (data.user.email) {
        this.addKeyValue('Email', data.user.email);
      }
      this.addSpace(10);
    }

    // Health Assessment
    if (data.assessment) {
      this.addSectionTitle('Health Assessment');

      // Risk score card
      const riskColor = data.assessment.risk === 'High Risk'
        ? COLORS.danger
        : data.assessment.risk === 'Moderate Risk'
          ? COLORS.warning
          : COLORS.success;

      this.addCard(
        'Cardiovascular Health Score',
        `${data.assessment.score}/100 - ${data.assessment.risk}`,
        riskColor
      );

      this.addKeyValue('Assessment Date', formatDate(data.assessment.date));
      this.addKeyValue('Blood Pressure', `${data.assessment.vitals.systolic} mmHg (systolic)`);
      this.addKeyValue('Total Cholesterol', `${data.assessment.vitals.cholesterol} mg/dL`);

      if (data.assessment.details) {
        this.addSpace(5);
        this.addText(data.assessment.details);
      }
      this.addSpace(10);
    }

    // Medications
    if (data.medications && data.medications.length > 0) {
      this.addSectionTitle('Current Medications');

      data.medications.forEach((med, index) => {
        this.checkPageBreak(15);
        this.addText(`${index + 1}. ${med.name}`, { bold: true });
        this.addKeyValue('   Dosage', med.dosage);
        this.addKeyValue('   Frequency', med.frequency);
        this.addKeyValue('   Time', med.times.join(', '));
        this.addSpace(3);
      });
      this.addSpace(5);
    }

    // Appointments
    if (data.appointments && data.appointments.length > 0) {
      this.addSectionTitle('Upcoming Appointments');

      data.appointments.forEach((apt, index) => {
        this.checkPageBreak(15);
        this.addText(`${index + 1}. Dr. ${apt.doctorName}`, { bold: true });
        this.addKeyValue('   Specialty', apt.specialty);
        this.addKeyValue('   Date', `${formatDate(apt.date)} at ${apt.time}`);
        this.addKeyValue('   Type', apt.type === 'video' ? 'Video Call' : 'In-Person');
        this.addKeyValue('   Location', apt.location);
        this.addSpace(3);
      });
    }

    // Biometric History
    if (data.biometricHistory && data.biometricHistory.length > 0) {
      this.addSectionTitle('Biometric History');

      // Table header
      this.checkPageBreak(30);
      this.doc!.setFont('helvetica', 'bold');
      this.doc!.setFontSize(9);
      this.doc!.setTextColor(...hexToRgb(COLORS.textLight));

      const colWidths = [40, 35, 35, 35, 25];
      const headers = ['Date', 'Systolic', 'Diastolic', 'Heart Rate', 'Weight'];
      let xPos = this.pageMargin;

      headers.forEach((header, i) => {
        this.doc!.text(header, xPos, this.currentY);
        xPos += colWidths[i];
      });

      this.currentY += 5;
      this.doc!.setDrawColor(...hexToRgb(COLORS.border));
      this.doc!.line(this.pageMargin, this.currentY, this.pageWidth - this.pageMargin, this.currentY);
      this.currentY += 3;

      // Table rows
      this.doc!.setFont('helvetica', 'normal');
      this.doc!.setTextColor(...hexToRgb(COLORS.text));

      data.biometricHistory.slice(0, 20).forEach((entry) => {
        this.checkPageBreak(8);
        xPos = this.pageMargin;

        const values = [
          formatDate(entry.date),
          entry.systolic?.toString() || '-',
          entry.diastolic?.toString() || '-',
          entry.heartRate?.toString() || '-',
          entry.weight ? `${entry.weight} kg` : '-',
        ];

        values.forEach((val, i) => {
          this.doc!.text(val, xPos, this.currentY);
          xPos += colWidths[i];
        });

        this.currentY += 6;
      });
    }

    this.addFooter();
    this.savePdf('cardio-health-report.pdf');
  }

  /**
   * Export chat history as PDF
   */
  exportChatHistory(data: ChatExport): void {
    this.initDocument('Chat History');

    this.addSectionTitle('Conversation Log');
    this.addKeyValue('Export Date', formatDateTime(data.exportDate));
    if (data.sessionId) {
      this.addKeyValue('Session ID', data.sessionId);
    }
    this.addKeyValue('Total Messages', data.messages.length.toString());
    this.addSpace(10);

    // Messages
    data.messages.forEach((msg) => {
      this.checkPageBreak(20);

      const isUser = msg.role === 'user';
      const bgColor = isUser ? '#E5E7EB' : '#F3F4F6';
      const labelColor = isUser ? COLORS.textLight : COLORS.primary;

      // Message header
      this.doc!.setFont('helvetica', 'bold');
      this.doc!.setFontSize(9);
      this.doc!.setTextColor(...hexToRgb(labelColor));
      this.doc!.text(
        isUser ? 'You' : 'Cardio AI',
        this.pageMargin,
        this.currentY
      );

      this.doc!.setFont('helvetica', 'normal');
      this.doc!.setTextColor(...hexToRgb(COLORS.textLight));
      this.doc!.text(
        formatDateTime(msg.timestamp),
        this.pageWidth - this.pageMargin,
        this.currentY,
        { align: 'right' }
      );

      this.currentY += 4;

      // Message content
      this.doc!.setFont('helvetica', 'normal');
      this.doc!.setFontSize(10);
      this.doc!.setTextColor(...hexToRgb(COLORS.text));

      const lines = this.doc!.splitTextToSize(msg.content || '', this.contentWidth - 10);
      const boxHeight = Math.max(10, lines.length * 4 + 6);

      // Message background
      this.doc!.setFillColor(...hexToRgb(bgColor));
      this.doc!.roundedRect(this.pageMargin, this.currentY - 2, this.contentWidth, boxHeight, 2, 2, 'F');

      // Message text
      this.doc!.text(lines, this.pageMargin + 5, this.currentY + 4);

      this.currentY += boxHeight + 5;
    });

    this.addFooter();
    this.savePdf('cardio-chat-history.pdf');
  }

  /**
   * Generate health summary for quick export
   */
  exportQuickSummary(): void {
    // Get data from localStorage
    const savedAssessment = localStorage.getItem('last_assessment');
    const savedMeds = localStorage.getItem('user_medications');
    const savedAppts = localStorage.getItem('user_appointments');

    const data: HealthReport = {
      assessment: savedAssessment ? JSON.parse(savedAssessment) : undefined,
      medications: savedMeds ? JSON.parse(savedMeds) : undefined,
      appointments: savedAppts ? JSON.parse(savedAppts) : undefined,
    };

    // Check if there's any data to export
    if (!data.assessment && !data.medications?.length && !data.appointments?.length) {
      // Return early - caller should show a toast/message
      console.warn('[PDFExport] No health data available for export');
      throw new Error('No health data available. Complete a health assessment first.');
    }

    this.exportHealthReport(data);
  }

  /**
   * Save PDF using Capacitor Filesystem on native, or browser download on web
   */
  private async savePdf(filename: string): Promise<void> {
    if (!this.doc) return;

    try {
      // Check if running on native platform
      const { Capacitor } = await import('@capacitor/core');
      if (Capacitor.isNativePlatform()) {
        try {
          const { Filesystem, Directory } = await import('@capacitor/filesystem');
          const pdfOutput = this.doc.output('datauristring');
          const base64Data = pdfOutput.split(',')[1];

          await Filesystem.writeFile({
            path: filename,
            data: base64Data,
            directory: Directory.Documents,
          });

          // Try to share the file
          try {
            const { Share } = await import('@capacitor/share');
            const fileUri = await Filesystem.getUri({
              path: filename,
              directory: Directory.Documents,
            });
            await Share.share({
              title: filename,
              url: fileUri.uri,
              dialogTitle: 'Share Health Report',
            });
          } catch {
            // Share not available, file is saved to Documents
            console.log(`[PDFExport] File saved to Documents/${filename}`);
          }
          return;
        } catch (err) {
          console.warn('[PDFExport] Filesystem save failed, falling back to download:', err);
        }
      }
    } catch {
      // Not running on Capacitor, use browser download
    }

    // Browser fallback
    this.doc.save(filename);
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

export const pdfExportService = new PDFExportService();

// Also export the class for testing
export { PDFExportService };

// Export types
export type { HealthReport, BiometricEntry, ChatExport };
