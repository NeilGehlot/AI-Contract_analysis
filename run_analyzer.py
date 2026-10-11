#!/usr/bin/env python3
"""Analyze one PDF contract and write JSON, text, and PDF reports.

Run from the repo root:
    python run_analyzer.py data/raw/sample_contract.pdf
"""

import sys
import argparse
from pathlib import Path
from colorama import init, Fore, Style

from src.analysis.analyzer import ContractAnalyzer
from src.analysis.report_generator import ReportGenerator

init()

def print_header():
    """Print application header"""
    print(Fore.CYAN + "="*80)
    print(Fore.CYAN + "CONTRACT ANALYZER - AI-Powered Contract Analysis & Risk Detection")
    print(Fore.CYAN + "Manipal University Jaipur - Semester 5 (ML Risk Classifier)")
    print(Fore.CYAN + "="*80)
    print(Style.RESET_ALL)

def print_summary(result: dict):
    """Print analysis summary to console"""
    print("\n" + Fore.GREEN + "="*80)
    print(Fore.GREEN + "ANALYSIS SUMMARY")
    print(Fore.GREEN + "="*80 + Style.RESET_ALL)
    
    print(f"\n{Fore.YELLOW}Contract Type:{Style.RESET_ALL} {result['classification']['type']}")
    print(f"{Fore.YELLOW}Confidence:{Style.RESET_ALL} {result['classification']['confidence']:.2%}")
    
    print(f"\n{Fore.YELLOW}Entities Found:{Style.RESET_ALL}")
    print(f"  - Parties: {len(result['entities']['parties'])}")
    print(f"  - Dates: {len(result['entities']['dates'])}")
    print(f"  - Amounts: {len(result['entities']['amounts'])}")
    
    print(f"\n{Fore.YELLOW}Risk Assessment:{Style.RESET_ALL}")
    flagged = [r for r in result['risks'] if r['severity'] in ('HIGH', 'MEDIUM')]
    if not flagged:
        print(f"  {Fore.GREEN}• No significant risks detected{Style.RESET_ALL}")
    for risk in flagged:
        color = Fore.RED if risk['severity'] == 'HIGH' else Fore.YELLOW
        topic = risk.get('topic') or 'Risk'
        confidence = risk.get('confidence')
        score = f" ({confidence:.0%})" if isinstance(confidence, float) else ""
        print(f"  {color}[{risk['severity']}] {topic}{score}{Style.RESET_ALL}")
        print(f"    Clause: {risk.get('clause', '')}")
        print(f"    Why: {risk.get('consideration') or risk.get('description', '')}")
        print(f"    What to do: {risk.get('recommendation', '')}")
    
    print("\n" + Fore.GREEN + "="*80 + Style.RESET_ALL)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze PDF contracts for risks and extract key information'
    )
    parser.add_argument(
        'pdf_file',
        help='Path to PDF contract file'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/reports',
        help='Output directory for reports (default: outputs/reports)'
    )
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF report generation (text and JSON only)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Generate only JSON output'
    )
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"{Fore.RED}Error: File not found: {pdf_path}{Style.RESET_ALL}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"{Fore.RED}Error: File must be a PDF{Style.RESET_ALL}")
        sys.exit(1)
    
    print_header()
    
    try:
        #analyzer
        print(f"\n{Fore.CYAN}Initializing analyzer...{Style.RESET_ALL}")
        analyzer = ContractAnalyzer()
        report_gen = ReportGenerator(args.output_dir)
        
        #contract
        print(f"\n{Fore.CYAN}Analyzing contract: {pdf_path.name}{Style.RESET_ALL}\n")
        result = analyzer.analyze(str(pdf_path))
        
        if not result['success']:
            print(f"{Fore.RED}Analysis failed: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")
            sys.exit(1)
        
        print_summary(result)
        
        print(f"\n{Fore.CYAN}Generating reports...{Style.RESET_ALL}")
        
        base_name = pdf_path.stem
        
        json_path = report_gen.save_json(result, f"{base_name}.json")
        print(f"  JSON: {json_path}")
        
        if not args.json_only:
            txt_path = report_gen.generate_text_report(result, f"{base_name}.txt")
            print(f"  Text: {txt_path}")
            
            if not args.no_pdf:
                pdf_report_path = report_gen.generate_pdf_report(result, f"{base_name}_report.pdf")
                print(f"  PDF: {pdf_report_path}")
        
        print(f"\n{Fore.GREEN}Analysis complete.{Style.RESET_ALL}")
        print(f"\nReports saved to: {Fore.CYAN}{args.output_dir}{Style.RESET_ALL}\n")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Analysis cancelled by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()