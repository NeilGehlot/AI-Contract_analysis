import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ReportGenerator:
    """
    Generate analysis reports
    """

    def _default_filename(self, prefix, extension):
        """
        Generate timestamp filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"


    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize report generator
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ReportGenerator initialized (output: {output_dir})")
    

    def generate_text_report(self, result: Dict, output_name: str = None) -> str:
        """
        Generate text report
        """
        if output_name is None:
            output_name = self._default_filename("report", "txt")
        
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as report_file:
            report_file.write(self._format_text_report(result))
        
        logger.info(f"Text report saved: {output_path}")
        return str(output_path)

    
    def generate_pdf_report(self, result: Dict, output_name: str = None) -> str:
        """
        Generate PDF report
        """
        if output_name is None:
            output_name = self._default_filename("report", "pdf")
        
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
        """
        Save results as JSON
        """
        if output_name is None:
            output_name = self._default_filename("report", "json")
        
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON saved: {output_path}")
        return str(output_path)

    
    def _split_risks(self, risks):
        """
        Splits the risks by severity
        """
        high_risks = [
            risk for risk in risks
            if risk["severity"] == "HIGH"
        ]
    
        medium_risks = [
            risk for risk in risks
            if risk["severity"] == "MEDIUM"
        ]
    
        return high_risks, medium_risks

    
    def _format_text_report(self, result: Dict) -> str:
        """
        Format text report
        """
        file_info = result["file_info"]
        classification = result["classification"]
        entities = result["entities"]

        lines = []
        lines.append("="*80)
        lines.append("CONTRACT ANALYSIS REPORT")
        lines.append("="*80)
        lines.append("")
        
        
        lines.append("FILE INFORMATION:")
        lines.append(f"  Filename: {file_info['filename']}")
        lines.append(f"  Pages: {file_info['pages']}")
        lines.append(f"  Analyzed: {file_info['analyzed_at']}")
        lines.append("")
        
        
        lines.append("CONTRACT TYPE:")
        lines.append(f"  Type: {classification['type']}")
        lines.append(f"  Confidence: {classification['confidence']:.2%}")
        lines.append("")
        
        
        lines.append("EXTRACTED ENTITIES:")
        lines.append("")
        
        lines.append("  Parties:")
        if entities['parties']:
            for i, party in enumerate(entities['parties'], 1):
                lines.append(f"    {i}. {party}")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Key Dates:")
        if entities['dates']:
            for date in entities['dates'][:5]:
                lines.append(f"    • {date['date']} ({date['type']})")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Monetary Amounts:")
        if entities['amounts']:
            for amt in entities['amounts'][:5]:
                lines.append(f"    • {amt['currency']}{amt['amount']}")
        else:
            lines.append("    None found")
        lines.append("")
        
        lines.append("  Contract Terms:")
        if entities['terms']:
            for key, value in entities['terms'].items():
                lines.append(f"    • {key.replace('_', ' ').title()}: {value}")
        else:
            lines.append("    None found")
        lines.append("")
        
        
        lines.append("RISK ASSESSMENT:")
        lines.append("")
        
        if result['risks']:
            high_risks,medium_risks = self._split_risks(result["risks"])
            
            if high_risks:
                lines.append(f"  HIGH RISKS ({len(high_risks)}):")
                for i, risk in enumerate(high_risks, 1):
                    lines.append(f"    {i}. {risk['description']}")
                    lines.append(f"       Recommendation: {risk['recommendation']}")
                lines.append("")
            
            if medium_risks:
                lines.append(f"  MEDIUM RISKS ({len(medium_risks)}):")
                for i, risk in enumerate(medium_risks, 1):
                    lines.append(f"    {i}. {risk['description']}")
                    lines.append(f"       Recommendation: {risk['recommendation']}")
                lines.append("")
        else:
            lines.append("  No significant risks detected")
        
        lines.append("="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)
    

    def _build_pdf_content(self, result: Dict) -> list:
        """
        Build PDF content
        """
        story = []
        styles = getSampleStyleSheet()
        
        file_info = result["file_info"]
        classification = result["classification"]
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        story.append(Paragraph("CONTRACT ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        
        story.append(Paragraph("<b>File Information</b>", styles['Heading2']))
        file_data = [
            ['Filename:', file_info['filename']],
            ['Pages:', str(file_info['pages'])],
            ['Analyzed:', file_info['analyzed_at']]
        ]
        file_table = Table(file_data, colWidths=[2*inch, 4*inch])
        file_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
        ]))
        story.append(file_table)
        story.append(Spacer(1, 0.3*inch))
        
        
        story.append(Paragraph("<b>Contract Type</b>", styles['Heading2']))
        story.append(Paragraph(
            f"Type: <b>{classification['type']}</b> "
            f"(Confidence: {classification['confidence']:.2%})",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        
        story.append(Paragraph("<b>Risk Assessment</b>", styles['Heading2']))
        if result['risks']:
            for risk in result['risks']:
                color = colors.red if risk['severity'] == 'HIGH' else colors.orange
                story.append(Paragraph(
                    f"<b><font color='{color}'>[{risk['severity']}]</font> {risk['description']}</b>",
                    styles['Normal']
                ))
                story.append(Paragraph(
                    f"Recommendation: {risk['recommendation']}",
                    styles['Normal']
                ))
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("No significant risks detected", styles['Normal']))
        
        return story