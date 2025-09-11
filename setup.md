# Rajasthan High Court Judgment Scraper

A comprehensive Python script to automatically download judgments from Rajasthan High Court with incremental daily updates and captcha solving capabilities.

## 🚀 Features

- **Incremental Downloading**: Only downloads new judgments, avoiding duplicates
- **Automated Captcha Solving**: Uses OCR to solve captchas automatically
- **Comprehensive Data Extraction**: Extracts all table information along with PDF downloads
- **State Management**: Tracks downloaded judgments to enable incremental updates
- **Error Handling**: Robust error handling with retry mechanisms
- **Bonus**: Supreme Court captcha solver

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Chrome/Chromium browser
- Tesseract OCR

### For Google Colab
```bash
!apt-get update
!apt-get install -y chromium-browser chromium-chromedriver tesseract-ocr
!pip install selenium pandas opencv-python pillow pytesseract requests numpy
```

### For Local Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Chrome/Chromium (Ubuntu/Debian)
sudo apt-get install chromium-browser chromium-chromedriver

# Install Tesseract OCR
sudo apt-get install tesseract-ocr

# For macOS
brew install chromedriver tesseract

# For Windows
# Download ChromeDriver from https://chromedriver.chromium.org/
# Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
```

## 🔧 Installation

1. **Clone or Download** the scraper script
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Setup ChromeDriver** (if not using Colab)
4. **Install Tesseract OCR**

## 📖 Usage

### Basic Usage
```python
from rajasthan_hc_scraper import RajasthanHCJudgmentScraper

# Initialize scraper
scraper = RajasthanHCJudgmentScraper()

# Run incremental scrape (last 10 days)
judgments = scraper.run_incremental_scrape()
```

### Custom Date Range
```python
# Scrape specific date range
judgments = scraper.scrape_judgments("01/09/2024", "11/09/2024")
scraper.save_to_csv(judgments)
```

### Jupyter Notebook
Use the provided notebook for interactive usage and testing.

## 📁 Output Structure

```
rajasthan_hc_judgments/
├── judgments.csv          # All judgment metadata
├── scraper_state.json     # State tracking file
└── pdfs/                  # Downloaded PDF files
    ├── judgment1.pdf
    ├── judgment2.pdf
    └── ...
```

## 📊 CSV Output Columns

The CSV file contains all information from the website table plus:
- `pdf_filename`: Name of downloaded PDF file
- `pdf_url`: Original URL of the PDF
- All original table columns (Case Number, Judgment Date, Judge Name, etc.)

## 🔄 Incremental Processing Logic

1. **First Run**: Downloads all judgments from last 10 days
2. **Subsequent Runs**: Only downloads new judgments that weren't downloaded before
3. **State Tracking**: Uses `scraper_state.json` to track downloaded judgments
4. **Unique Identification**: Creates unique IDs based on case number, date, and judge

## 🎯 Captcha Solving

### Automated Solving
- Uses OpenCV for image preprocessing
- Applies Tesseract OCR for text recognition
- Implements retry mechanism (3 attempts)
- Preprocessing includes denoising, thresholding, and morphological operations

### Manual Fallback
If automated solving fails, you can modify the `solve_captcha` method:

```python
def solve_captcha_manual(self, captcha_image_element) -> str:
    captcha_image_element.screenshot("captcha_display.png")
    # Display image and get manual input
    return input("Enter captcha: ").strip()
```

## 🎁 Bonus: Supreme Court Captcha Solver

```python
from rajasthan_hc_scraper import SCICaptchaSolver

solver = SCICaptchaSolver()
result = solver.solve_sci_captcha("captcha_image.png")
```

## ⚙️ Configuration Options

### Scraper Configuration
```python
scraper = RajasthanHCJudgmentScraper(
    download_dir="custom_directory"  # Custom download directory
)

# Modify Chrome options
scraper.chrome_options.add_argument("--no-headless")  # Show browser
```