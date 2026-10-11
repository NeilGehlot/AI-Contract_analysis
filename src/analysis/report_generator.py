"""
Report Generator
Generates analysis reports in text and PDF format
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _risk_lines(index: int, risk: dict) -> list[str]:
    topic = risk.get("topic") or "Risk"
    clause = risk.get("clause") or ""
    why = risk.get("consideration") or risk.get("description") or ""
    action = risk.get("recommendation") or ""
    confidence = risk.get("confidence")
    score = f" ({confidence:.0%})" if isinstance(confidence, float) else ""
    return [
        f"    {index}. {topic}{score}",
        f"       Clause: {clause}",
        f"       Why: {why}",
        f"       What to do: {action}",
    ]

class ReportGenerator:
    """Generate analysis reports"""
    
    def __init__(self, output_dir: str = "outputs/reports"):
        """Initialize report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ReportGenerator initialized (output: {output_dir})")
    
    def generate_text_report(self, result: Dict, output_name: str = None) -> str:
        """
        Generate text report
        
        Args:
            result: Analysis result
            output_name: Output filename
            
        Returns:
            Path to generated report
        """
        if output_name is None:
            output_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report(result))
        
        logger.info(f"Text report saved: {output_path}")
        return str(output_path)
    
    def generate_pdf_report(self, result: Dict, output_name: str = None) -> str:
        """
        Generate PDF report
        
        Args:
            result: Analysis result
            output_name: Output filename
            
        Returns:
            Path to generated report
        """
        if output_name is None:
            output_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        output_path = self.output_dir / output_name
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = self._build_pdf_content(result)
        doc.build(story)
        
        logger.info(f"PDF report saved: {output_path}")
        return str(output_path)
    
    def save_json(self, result: Dict, output_name: str = None) -> str:
        """Save results as JSON"""
        if output_name is None:
            output_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON saved: {output_path}")
        return str(output_path)
    
    def _format_text_report(self, result: Dict) -> str:
        """Format text report"""
        lines = []
        lines.append("="*80)
        lines.append("CONTRACT ANALYSIS REPORT")
        lines.append("="*80)
        lines.append("")
        
        # File info
        lines.append("FILE INFORMATION:")
        lines.append(f"  Filename: {result['file_info']['filename']}")
        lines.append(f"  Pages: {result['file_info']['pages']}")
        lines.append(f"  Analyzed: {result['file_info']['analyzed_at']}")
        lines.append("")
        
        # Classification
        lines.append("CONTRACT TYPE:")
        lines.append(f"  Type: {result['classification']['type']}")
        lines.append(f"  Confidence: {result['classification']['confidence']:.2%}")
        lines.append("")
        
        # Entities
        lines.append("EXTRACTED ENTITIES:")
        lines.append("")
        
        lines.append("  Parties:")
        if result['entities']['parties']:
            for i, party in enumerate(result['entities']['parties'], 1):
                lines.append(f"    {i}. {party}")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Key Dates:")
        if result['entities']['dates']:
            for date in result['entities']['dates'][:5]:
                if isinstance(date, dict):
                    lines.append(f"    • {date.get('date', date)} ({date.get('type', 'date')})")
                else:
                    lines.append(f"    • {date}")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Monetary Amounts:")
        if result['entities']['amounts']:
            for amt in result['entities']['amounts'][:5]:
                if isinstance(amt, dict):
                    lines.append(f"    • {amt.get('currency', '')}{amt.get('amount', amt)}")
                else:
                    lines.append(f"    • {amt}")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Contract Terms:")
        if result['entities']['terms']:
            terms = result['entities']['terms']
            if isinstance(terms, dict):
                for key, value in terms.items():
                    lines.append(f"    • {key.replace('_', ' ').title()}: {value}")
            else:
                lines.append(f"    • {terms}")
        else:
            lines.append("    None found")
        lines.append("")
        
        # Risks
        lines.append("RISK ASSESSMENT:")
        lines.append("")
        
        if result['risks']:
            high_risks = [r for r in result['risks'] if r['severity'] == 'HIGH']
            medium_risks = [r for r in result['risks'] if r['severity'] == 'MEDIUM']
            
            if high_risks:
                lines.append(f"  HIGH RISKS ({len(high_risks)}):")
                for i, risk in enumerate(high_risks, 1):
                    lines.extend(_risk_lines(i, risk))
                lines.append("")
            
            if medium_risks:
                lines.append(f"  MEDIUM RISKS ({len(medium_risks)}):")
                for i, risk in enumerate(medium_risks, 1):
                    lines.extend(_risk_lines(i, risk))
                lines.append("")
        else:
            lines.append("  No significant risks detected")
        
        lines.append("="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def _build_pdf_content(self, result: Dict) -> list:
        """Build PDF content"""
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        story.append(Paragraph("CONTRACT ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # File info
        story.append(Paragraph("<b>File Information</b>", styles['Heading2']))
        file_data = [
            ['Filename:', result['file_info']['filename']],
            ['Pages:', str(result['file_info']['pages'])],
            ['Analyzed:', result['file_info']['analyzed_at']]
        ]
        file_table = Table(file_data, colWidths=[2*inch, 4*inch])
        file_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
        ]))
        story.append(file_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Classification
        story.append(Paragraph("<b>Contract Type</b>", styles['Heading2']))
        story.append(Paragraph(
            f"Type: <b>{result['classification']['type']}</b> "
            f"(Confidence: {result['classification']['confidence']:.2%})",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Risks
        story.append(Paragraph("<b>Risk Assessment</b>", styles['Heading2']))
        if result['risks']:
            for risk in result['risks']:
                color = colors.red if risk['severity'] == 'HIGH' else colors.orange
                topic = escape(str(risk.get("topic") or "Risk"))
                clause = escape(str(risk.get("clause") or ""))
                why = escape(str(risk.get("consideration") or risk.get("description") or ""))
                action = escape(str(risk.get("recommendation") or ""))
                confidence = risk.get("confidence")
                score = f" ({confidence:.0%})" if isinstance(confidence, float) else ""
                story.append(Paragraph(
                    f"<b><font color='{color}'>[{risk['severity']}]</font> {topic}{score}</b>",
                    styles['Normal']
                ))
                story.append(Paragraph(f"Clause: {clause}", styles['Normal']))
                story.append(Paragraph(f"Why: {why}", styles['Normal']))
                story.append(Paragraph(f"What to do: {action}", styles['Normal']))
                story.append(Spacer(1, 0.12*inch))
        else:
            story.append(Paragraph("No significant risks detected", styles['Normal']))
        
        return story