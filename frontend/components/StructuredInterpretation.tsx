import React, { useState, useMemo } from 'react';

interface Section {
    title: string;
    content: string;
    items?: string[];
    testAnalyses?: TestAnalysis[];
}

interface TestAnalysis {
    name: string;
    value: string;
    status: string;
    risk: string;
    explanation: string;
}

/**
 * Parse raw LLM markdown into structured sections.
 */
function parseClinicalInterpretation(raw: string): Section[] {
    const sections: Section[] = [];
    if (!raw) return sections;

    // Normalize line endings
    const text = raw.replace(/\r\n/g, '\n').trim();

    // Split by markdown ## / ### headers OR bold headers: **Header:**
    const headerRegex = /(?:^|\n)(?:#{1,3}\s+(.+)|(?:\*\*([^*]+?)(?::)?\*\*\s*))/g;
    const parts: { title: string; body: string; startIndex: number }[] = [];
    let match: RegExpExecArray | null;

    while ((match = headerRegex.exec(text)) !== null) {
        const title = (match[1] || match[2]).trim().replace(/:$/, '').replace(/\*\*/g, '');
        parts.push({ title, body: '', startIndex: match.index + match[0].length });
    }

    // Fill in body content between headers
    for (let i = 0; i < parts.length; i++) {
        const nextStart = i + 1 < parts.length ? parts[i + 1].startIndex - (parts[i + 1].title.length + 10) : text.length;
        // Find the actual start of the next header in the raw text
        const bodyEnd = i + 1 < parts.length
            ? text.lastIndexOf('\n', text.indexOf(parts[i + 1].title, parts[i].startIndex))
            : text.length;
        parts[i].body = text.slice(parts[i].startIndex, bodyEnd > parts[i].startIndex ? bodyEnd : nextStart).trim();
    }

    // If no headers found, return single section
    if (parts.length === 0) {
        return [{ title: 'Clinical Summary', content: text.replace(/\*\*/g, '') }];
    }

    for (const part of parts) {
        const section: Section = { title: part.title, content: '' };

        // Check for bullet points
        const bulletLines = part.body.split('\n').filter(l => l.trim().startsWith('*') || l.trim().startsWith('-') || l.trim().match(/^\d+\./));
        const nonBulletLines = part.body.split('\n').filter(l => !l.trim().startsWith('*') && !l.trim().startsWith('-') && !l.trim().match(/^\d+\./) && l.trim());

        if (part.title.toLowerCase().includes('test-by-test') || part.title.toLowerCase().includes('test by test')) {
            // Parse test analysis items
            section.testAnalyses = [];
            const testRegex = /\*\s*\*\*([^*]+)\*\*[:\s]*([^(]*)\(([^)]+)\)\s*[—–-]\s*Status:\s*([^,]+),\s*Risk:\s*([^.]+)\.\s*(.*)/g;
            let testMatch: RegExpExecArray | null;
            while ((testMatch = testRegex.exec(part.body)) !== null) {
                section.testAnalyses.push({
                    name: testMatch[1].trim(),
                    value: testMatch[2].trim(),
                    status: testMatch[4].trim(),
                    risk: testMatch[5].trim(),
                    explanation: testMatch[6].trim(),
                });
            }
            // Fallback: simpler bullet parse
            if (section.testAnalyses.length === 0) {
                for (const line of bulletLines) {
                    const cleaned = line.replace(/^\s*[\*\-]\s*/, '').replace(/^\d+\.\s*/, '').trim();
                    const simpleMatch = cleaned.match(/\*\*([^*]+)\*\*[:\s]*(.*)/);
                    if (simpleMatch) {
                        section.testAnalyses.push({
                            name: simpleMatch[1].trim(),
                            value: '',
                            status: '',
                            risk: '',
                            explanation: simpleMatch[2].trim().replace(/\*\*/g, ''),
                        });
                    }
                }
            }
        } else if (bulletLines.length > 0) {
            section.content = nonBulletLines.join(' ').replace(/\*\*/g, '').trim();
            section.items = bulletLines.map(l =>
                l.replace(/^\s*[\*\-]\s*/, '').replace(/^\d+\.\s*/, '').replace(/\*\*/g, '').trim()
            ).filter(Boolean);
        } else {
            section.content = part.body.replace(/\*\*/g, '').trim();
        }

        sections.push(section);
    }

    return sections;
}

// ============================================================================
// Visual Helpers
// ============================================================================

interface SectionStyle {
    icon: string;
    iconColor: string;
    gradientFrom: string;
    gradientTo: string;
    accentBorder: string;
    accentBg: string;
    badgeColor: string;
}

function getSectionStyle(title: string): SectionStyle {
    const t = title.toLowerCase();
    if (t.includes('why') || t.includes('prediction') || t.includes('summary') || t.includes('overview'))
        return { icon: 'psychology', iconColor: 'text-violet-500', gradientFrom: 'from-violet-500/10', gradientTo: 'to-purple-500/5', accentBorder: 'border-l-violet-500', accentBg: 'bg-violet-50 dark:bg-violet-950/20', badgeColor: 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300' };
    if (t.includes('test'))
        return { icon: 'biotech', iconColor: 'text-blue-500', gradientFrom: 'from-blue-500/10', gradientTo: 'to-cyan-500/5', accentBorder: 'border-l-blue-500', accentBg: 'bg-blue-50 dark:bg-blue-950/20', badgeColor: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' };
    if (t.includes('key risk') || t.includes('risk factor') || t.includes('concern'))
        return { icon: 'warning', iconColor: 'text-red-500', gradientFrom: 'from-red-500/10', gradientTo: 'to-orange-500/5', accentBorder: 'border-l-red-500', accentBg: 'bg-red-50 dark:bg-red-950/20', badgeColor: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300' };
    if (t.includes('protective') || t.includes('positive') || t.includes('strength'))
        return { icon: 'verified_user', iconColor: 'text-emerald-500', gradientFrom: 'from-emerald-500/10', gradientTo: 'to-green-500/5', accentBorder: 'border-l-emerald-500', accentBg: 'bg-emerald-50 dark:bg-emerald-950/20', badgeColor: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300' };
    if (t.includes('recommendation') || t.includes('action') || t.includes('next step') || t.includes('advice'))
        return { icon: 'task_alt', iconColor: 'text-teal-500', gradientFrom: 'from-teal-500/10', gradientTo: 'to-emerald-500/5', accentBorder: 'border-l-teal-500', accentBg: 'bg-teal-50 dark:bg-teal-950/20', badgeColor: 'bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300' };
    if (t.includes('risk summary') || t.includes('overall') || t.includes('conclusion'))
        return { icon: 'assessment', iconColor: 'text-amber-500', gradientFrom: 'from-amber-500/10', gradientTo: 'to-orange-500/5', accentBorder: 'border-l-amber-500', accentBg: 'bg-amber-50 dark:bg-amber-950/20', badgeColor: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300' };
    if (t.includes('clinical') || t.includes('medical') || t.includes('diagnosis'))
        return { icon: 'stethoscope', iconColor: 'text-sky-500', gradientFrom: 'from-sky-500/10', gradientTo: 'to-blue-500/5', accentBorder: 'border-l-sky-500', accentBg: 'bg-sky-50 dark:bg-sky-950/20', badgeColor: 'bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300' };
    if (t.includes('lifestyle') || t.includes('diet') || t.includes('exercise'))
        return { icon: 'directions_run', iconColor: 'text-green-500', gradientFrom: 'from-green-500/10', gradientTo: 'to-lime-500/5', accentBorder: 'border-l-green-500', accentBg: 'bg-green-50 dark:bg-green-950/20', badgeColor: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' };
    if (t.includes('medication') || t.includes('drug') || t.includes('treatment') || t.includes('prescription'))
        return { icon: 'medication', iconColor: 'text-purple-500', gradientFrom: 'from-purple-500/10', gradientTo: 'to-fuchsia-500/5', accentBorder: 'border-l-purple-500', accentBg: 'bg-purple-50 dark:bg-purple-950/20', badgeColor: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300' };
    if (t.includes('note') || t.includes('important') || t.includes('disclaimer') || t.includes('caveat'))
        return { icon: 'info', iconColor: 'text-slate-500', gradientFrom: 'from-slate-500/10', gradientTo: 'to-gray-500/5', accentBorder: 'border-l-slate-400', accentBg: 'bg-slate-50 dark:bg-slate-800/40', badgeColor: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400' };
    return { icon: 'article', iconColor: 'text-slate-500', gradientFrom: 'from-slate-500/5', gradientTo: 'to-slate-500/0', accentBorder: 'border-l-slate-300', accentBg: 'bg-slate-50 dark:bg-slate-800/40', badgeColor: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400' };
}

function getStatusColor(status: string): { bg: string; text: string; dot: string } {
    const s = status.toLowerCase();
    if (s.includes('normal')) return { bg: 'bg-green-50 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-300', dot: 'bg-green-500' };
    if (s.includes('borderline')) return { bg: 'bg-amber-50 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-300', dot: 'bg-amber-500' };
    if (s.includes('critical')) return { bg: 'bg-red-50 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-300', dot: 'bg-red-500' };
    if (s.includes('abnormal')) return { bg: 'bg-orange-50 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-300', dot: 'bg-orange-500' };
    return { bg: 'bg-slate-50 dark:bg-slate-800/50', text: 'text-slate-600 dark:text-slate-400', dot: 'bg-slate-400' };
}

function getRiskColor(risk: string): { bg: string; text: string } {
    const r = risk.toLowerCase();
    if (r.includes('low')) return { bg: 'bg-blue-50 dark:bg-blue-900/20', text: 'text-blue-600 dark:text-blue-300' };
    if (r.includes('moderate') || r.includes('medium')) return { bg: 'bg-yellow-50 dark:bg-yellow-900/20', text: 'text-yellow-700 dark:text-yellow-300' };
    if (r.includes('high') || r.includes('elevated')) return { bg: 'bg-red-50 dark:bg-red-900/20', text: 'text-red-600 dark:text-red-300' };
    return { bg: 'bg-slate-50 dark:bg-slate-800/50', text: 'text-slate-600 dark:text-slate-400' };
}

// ============================================================================
// Sub-Components
// ============================================================================

/** Collapsible section wrapper */
const CollapsibleSection: React.FC<{
    section: Section;
    style: SectionStyle;
    defaultOpen?: boolean;
    index: number;
    children: React.ReactNode;
}> = ({ section, style, defaultOpen = true, index, children }) => {
    const [open, setOpen] = useState(defaultOpen);

    return (
        <div
            className={`rounded-2xl border border-slate-100 dark:border-slate-700/40 overflow-hidden transition-all duration-300 ${open ? 'shadow-sm' : 'shadow-none'}`}
            style={{ animationDelay: `${index * 80}ms` }}
        >
            {/* Section header — clickable */}
            <button
                onClick={() => setOpen(!open)}
                className={`w-full flex items-center gap-3 px-4 py-3.5 text-left transition-colors border-l-[3px] ${style.accentBorder} ${style.accentBg} hover:brightness-95 dark:hover:brightness-110`}
            >
                <div className={`w-8 h-8 rounded-xl bg-gradient-to-br ${style.gradientFrom} ${style.gradientTo} flex items-center justify-center shrink-0 border border-white/60 dark:border-white/10`}>
                    <span className={`material-symbols-outlined text-[18px] ${style.iconColor}`}>{style.icon}</span>
                </div>
                <h4 className="flex-1 font-semibold text-[13px] text-slate-800 dark:text-slate-100 tracking-wide">
                    {section.title}
                </h4>
                {section.items && section.items.length > 0 && (
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${style.badgeColor}`}>
                        {section.items.length}
                    </span>
                )}
                {section.testAnalyses && section.testAnalyses.length > 0 && (
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${style.badgeColor}`}>
                        {section.testAnalyses.length} tests
                    </span>
                )}
                <span className={`material-symbols-outlined text-slate-400 text-lg transition-transform duration-300 ${open ? 'rotate-180' : ''}`}>
                    expand_more
                </span>
            </button>

            {/* Content — animated */}
            <div className={`transition-all duration-300 ease-in-out overflow-hidden ${open ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'}`}>
                <div className="p-4 bg-white/60 dark:bg-slate-900/30">
                    {children}
                </div>
            </div>
        </div>
    );
};

/** Test analysis card */
const TestCard: React.FC<{ test: TestAnalysis; index: number }> = ({ test, index }) => {
    const status = getStatusColor(test.status);
    const risk = getRiskColor(test.risk);

    return (
        <div className="group bg-white dark:bg-slate-800/70 rounded-xl border border-slate-100 dark:border-slate-700/30 p-3.5 transition-all duration-200 hover:shadow-md hover:border-slate-200 dark:hover:border-slate-600/50">
            <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                    <div className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 text-[11px] font-bold ${status.bg} ${status.text}`}>
                        {index + 1}
                    </div>
                    <span className="font-semibold text-[13px] text-slate-800 dark:text-slate-100 truncate">{test.name}</span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                    {test.status && (
                        <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${status.bg} ${status.text}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`}></span>
                            {test.status}
                        </span>
                    )}
                    {test.risk && (
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${risk.bg} ${risk.text}`}>
                            {test.risk}
                        </span>
                    )}
                </div>
            </div>
            {test.value && (
                <div className="mt-2 ml-8.5 pl-0.5">
                    <span className="text-xs text-slate-500 dark:text-slate-400">Value: </span>
                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">{test.value}</span>
                </div>
            )}
            {test.explanation && (
                <p className="mt-1.5 ml-8.5 pl-0.5 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                    {test.explanation}
                </p>
            )}
        </div>
    );
};

/** Bullet item row */
const BulletItem: React.FC<{ item: string; index: number; style: SectionStyle }> = ({ item, index, style }) => (
    <div className="flex items-start gap-3 group">
        <div className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 mt-0.5 text-[10px] font-bold border transition-colors ${style.badgeColor} border-transparent group-hover:border-current/20`}>
            {index + 1}
        </div>
        <p className="text-[13px] text-slate-700 dark:text-slate-300 leading-relaxed flex-1">
            {item}
        </p>
    </div>
);

// ============================================================================
// Main Component
// ============================================================================

interface Props {
    interpretation: string;
    isGrounded?: boolean;
}

const StructuredInterpretation: React.FC<Props> = ({ interpretation, isGrounded }) => {
    const sections = useMemo(() => parseClinicalInterpretation(interpretation), [interpretation]);
    const [expandedTests, setExpandedTests] = useState(false);

    if (sections.length === 0) return null;

    return (
        <div className="space-y-3.5">
            {/* Header */}
            <div className="flex items-center gap-3 pb-1">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/20">
                    <span className="material-symbols-outlined text-white text-lg">neurology</span>
                </div>
                <div className="flex-1">
                    <h3 className="font-bold text-sm text-slate-800 dark:text-white flex items-center gap-2">
                        AI Clinical Interpretation
                        {isGrounded && (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 text-[10px] font-bold rounded-full">
                                <span className="material-symbols-outlined text-[10px]">verified</span>
                                Grounded
                            </span>
                        )}
                    </h3>
                    <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
                        Powered by medical AI analysis
                    </p>
                </div>
            </div>

            {/* Divider */}
            <div className="h-px bg-gradient-to-r from-transparent via-slate-200 dark:via-slate-700 to-transparent" />

            {/* Sections */}
            {sections.map((section, idx) => {
                const sStyle = getSectionStyle(section.title);

                return (
                    <CollapsibleSection
                        key={idx}
                        section={section}
                        style={sStyle}
                        defaultOpen={idx < 3}
                        index={idx}
                    >
                        {/* Main content paragraph */}
                        {section.content && (
                            <p className="text-[13px] text-slate-600 dark:text-slate-300 leading-[1.7] mb-3">
                                {section.content}
                            </p>
                        )}

                        {/* Bullet items */}
                        {section.items && section.items.length > 0 && (
                            <div className="space-y-2.5">
                                {section.items.map((item, i) => (
                                    <BulletItem key={i} item={item} index={i} style={sStyle} />
                                ))}
                            </div>
                        )}

                        {/* Test-by-test analysis */}
                        {section.testAnalyses && section.testAnalyses.length > 0 && (
                            <div className="space-y-2">
                                {(expandedTests ? section.testAnalyses : section.testAnalyses.slice(0, 4)).map((test, i) => (
                                    <TestCard key={i} test={test} index={i} />
                                ))}

                                {section.testAnalyses.length > 4 && (
                                    <button
                                        onClick={() => setExpandedTests(!expandedTests)}
                                        className="w-full py-2.5 mt-1 text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 transition-colors flex items-center justify-center gap-1.5 rounded-xl hover:bg-indigo-50 dark:hover:bg-indigo-900/20"
                                    >
                                        <span className="material-symbols-outlined text-sm">
                                            {expandedTests ? 'unfold_less' : 'unfold_more'}
                                        </span>
                                        {expandedTests
                                            ? 'Show Fewer Tests'
                                            : `View All ${section.testAnalyses.length} Test Results`
                                        }
                                    </button>
                                )}
                            </div>
                        )}
                    </CollapsibleSection>
                );
            })}

            {/* Footer */}
            <div className="flex items-center justify-center pt-1">
                <p className="text-[10px] text-slate-400 dark:text-slate-600 flex items-center gap-1">
                    <span className="material-symbols-outlined text-[11px]">info</span>
                    AI-generated analysis — consult your physician for medical decisions
                </p>
            </div>
        </div>
    );
};

export default StructuredInterpretation;
