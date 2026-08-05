# Contract Analyzer - Semester 3 Prototype

AI-Powered Contract Analysis & Risk Detection Platform for Indian SMEs

## Features

- ✅ PDF text extraction (with OCR fallback)
- ✅ Entity extraction (parties, dates, amounts, terms)
- ✅ Contract type classification
- ✅ Risk detection (unlimited liability, long payment terms, missing clauses)
- ✅ Automated report generation (Text, PDF, JSON)
- ✅ Optimized for Indian Contract Act

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone/Download the project**
```bash
cd contract-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage
```bash
python run_analyzer.py data/raw/sample_contract.pdf
```

### Advanced Options
```bash
# Generate only JSON output
python run_analyzer.py contract.pdf --json-only

# Skip PDF report generation
python run_analyzer.py contract.pdf --no-pdf

# Custom output directory
python run_analyzer.py contract.pdf --output-dir my_reports
```

### Example Output