# Rajasthan High Court Judgment Scraper

This project implements a web scraper for downloading judgments from the Rajasthan High Court website. It features incremental daily downloads, captcha solving, and CSV output with relevant judgment information.

## Features

- **Incremental Daily Download**: Downloads judgments from the last 10 days by default, and only downloads new judgments on subsequent runs
- **Captcha Solving**: Uses OCR with image preprocessing to solve captchas automatically
- **CSV Output**: Generates a CSV file with all relevant judgment information
- **PDF Download**: Downloads judgment PDFs and stores them locally
- **Error Handling**: Robust error handling for network issues, captcha failures, and other potential problems
- **State Management**: Keeps track of downloaded judgments to avoid duplicates

## Requirements

See `requirements.txt` for a complete list of dependencies. The main requirements are:

- Python 3.7+
- Selenium
- pandas
- OpenCV
- Tesseract OCR
- Pillow

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (required for captcha solving):
   - Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### Basic Usage

Run the main script to download judgments from the last 10 days:

```bash
python main.py
```

### Advanced Usage

The script accepts several command-line arguments:

```bash
python main.py --days 30 --output-dir custom_output_dir --test-sci-captcha
```

- `--days`: Number of days to look back (default: 10)
- `--output-dir`: Output directory for judgments and CSV (default: rajasthan_hc_judgments)
- `--test-sci-captcha`: Test the Supreme Court of India captcha solver

## Testing

Run the test script to verify the functionality:

```bash
python test_scraper.py
```

## Project Structure

- `RajasthanHighCourtJudgmentScraper.py`: Main scraper implementation
- `Advanced-OCR-Captcha-Solver.py`: Advanced captcha solving implementation
- `main.py`: Entry point for the scraper
- `test_scraper.py`: Test script for verifying functionality
- `requirements.txt`: List of dependencies

## Output

The scraper generates the following outputs:

- CSV file with judgment information
- PDF files of judgments
- State file to track downloaded judgments

## Bonus: Supreme Court of India Captcha Solver

The project includes a bonus implementation of a captcha solver for the Supreme Court of India website. This solver uses advanced image preprocessing and OCR techniques to solve captchas offline.

## License

This project is licensed under the MIT License - see the LICENSE file for details.