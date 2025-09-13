# Rajasthan High Court Judgment Scraper - Type-Safe Version
# Fixed: All Pylance type warnings and global variable issues

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time, logging, os, re, requests, pytz
from datetime import datetime, timedelta
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import base64
from io import BytesIO
import cv2, numpy as np
import bs4
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
from typing import List, Tuple, Optional, Union

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters"
OUTPUT_DIR = "output"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "rajasthan_hc_judgments.csv")
MAX_CAPTCHA_RETRIES = 15

# Date range setup
ist = pytz.timezone('Asia/Kolkata')
current_ist_date = datetime.now(ist)
from_date_obj = current_ist_date - timedelta(days=10)
from_date_str = from_date_obj.strftime("%d/%m/%Y")
to_date_str = current_ist_date.strftime("%d/%m/%Y")

class RajasthanHCScraper:
    """Main scraper class to handle all operations"""
    
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.all_judgments: List[List[str]] = []
        self.downloaded_pdfs: set = set()
    
    def setup_driver(self) -> bool:
        """Setup Chrome driver with proper configuration"""
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass
        
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # PDF download preferences
        prefs = {
            "download.default_directory": os.path.abspath(PDF_DIR),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=chrome_options
            )
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, 20)
            logger.info("Chrome driver setup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            return False

    def preprocess_captcha_image(self, image: Image.Image) -> List[np.ndarray]:
        """Enhanced image preprocessing for better OCR accuracy"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Apply multiple preprocessing techniques
        processed_images = []
        
        # Method 1: Simple thresholding
        _, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        processed_images.append(thresh1)
        
        # Method 2: Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(thresh2)
        
        # Method 3: Otsu's thresholding
        _, thresh3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh3)
        
        # Method 4: Blur and threshold
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, thresh4 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        processed_images.append(thresh4)
        
        # Method 5: Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(cleaned)
        
        return processed_images

    def solve_captcha_enhanced(self, captcha_image: Image.Image, attempt_num: int = 1) -> Optional[str]:
        """Enhanced captcha solving with multiple OCR configurations"""
        
        # Get multiple preprocessed versions
        processed_images = self.preprocess_captcha_image(captcha_image)
        
        # OCR configurations - more specific for digits
        ocr_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789',
            r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789',
            r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789',
        ]
        
        # Try different combinations
        results = []
        
        for i, processed_img in enumerate(processed_images):
            pil_img = Image.fromarray(processed_img)
            
            # Scale up the image for better OCR (important for small captchas)
            width, height = pil_img.size
            if width < 200 or height < 50:
                scale_factor = max(200/width, 50/height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
            for config in ocr_configs:
                try:
                    text = pytesseract.image_to_string(pil_img, config=config)
                    digits = ''.join(filter(str.isdigit, text))
                    if len(digits) == 6:  # Expected captcha length
                        results.append((digits, i, config))
                    elif len(digits) > 6:
                        # Sometimes OCR returns extra digits, try to find 6 consecutive ones
                        for start in range(len(digits) - 5):
                            candidate = digits[start:start+6]
                            if len(candidate) == 6:
                                results.append((candidate, i, config))
                except:
                    continue
        
        # Return the most frequent result, or first valid one
        if results:
            # Count frequency of results
            result_counts = {}
            for result, _, _ in results:
                result_counts[result] = result_counts.get(result, 0) + 1
            
            # Return most frequent result
            best_result = max(result_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Captcha solved: {best_result} (found {result_counts[best_result]} times out of {len(results)} attempts)")
            return best_result
        
        logger.warning("Could not solve captcha with any method")
        return None

    def solve_captcha_with_retry(self) -> bool:
        """Main captcha solving function with intelligent retry logic"""
        
        if not self.driver or not self.wait:
            logger.error("Driver not initialized")
            return False
        
        for attempt in range(1, MAX_CAPTCHA_RETRIES + 1):
            try:
                logger.info(f"Captcha attempt {attempt}/{MAX_CAPTCHA_RETRIES}")
                
                # Refresh captcha image
                try:
                    # refresh_btn = self.driver.find_element(By.ID, "change-image")
                    # self.driver.execute_script("arguments[0].click();", refresh_btn)
                    # time.sleep(1)
                    refresh_btn = self.driver.find_element(By.ID, "change-image")
                    self.driver.execute_script("arguments[0].click();", refresh_btn)
                    time.sleep(2)  # Wait longer for new captcha to load
                except:
                    logger.warning("Could not refresh captcha image")
                
                # Wait a bit more on later attempts
                time.sleep(2 + (attempt * 0.3))  # Start with longer base wait
                
                # Get captcha image
                captcha_element = self.wait.until(EC.presence_of_element_located((By.ID, "captcha")))
                captcha_src = captcha_element.get_attribute("src")
                
                if not captcha_src:
                    logger.error("No captcha source found")
                    continue
                
                # Download captcha image
                if captcha_src.startswith("data:"):
                    captcha_base64 = captcha_src.split(",", 1)[1]
                    captcha_image = Image.open(BytesIO(base64.b64decode(captcha_base64)))
                else:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Referer': BASE_URL
                    }
                    resp = requests.get(captcha_src, headers=headers, timeout=10)
                    captcha_image = Image.open(BytesIO(resp.content))
                
                # Solve captcha
                solved = self.solve_captcha_enhanced(captcha_image, attempt)
                
                if not solved:
                    logger.warning(f"Attempt {attempt}: Could not solve captcha")
                    continue
                
                logger.info(f"Attempt {attempt}: Solved captcha as '{solved}'")
                
                # Clear and enter captcha
                captcha_input = self.driver.find_element(By.ID, "txtCaptcha")
                # captcha_input.clear()
                # time.sleep(0.5)
                # captcha_input.send_keys(solved)
                captcha_input.clear()
                time.sleep(1)
                # Ensure the field is focused
                self.driver.execute_script("arguments[0].focus();", captcha_input)
                captcha_input.send_keys(solved)
                time.sleep(0.5)  # Let the input register
                # Submit form
                # submit_btn = self.driver.find_element(By.ID, "btncasedetail1_1")
                # self.driver.execute_script("arguments[0].click();", submit_btn)
                submit_btn = self.driver.find_element(By.ID, "btncasedetail1_1")
                time.sleep(1)  # Give time for captcha to register
                submit_btn.click()  # Try normal click first
                # Wait for results or error
                time.sleep(3)
                
                # Check if we got results
                try:
                    # Look for results table or any indication of success
                    WebDriverWait(self.driver, 5).until(EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "table")),
                        EC.presence_of_element_located((By.ID, "div_datatable")),
                        EC.presence_of_element_located((By.CLASS_NAME, "dataTables_wrapper"))
                    ))
                    
                    # Additional check - make sure we're not still on the form page
                    current_url = self.driver.current_url
                    if "JudgementFilters" in current_url and self.driver.find_elements(By.ID, "txtCaptcha"):
                        # Still on form page, captcha was wrong
                        logger.warning(f"Attempt {attempt}: Still on form page, captcha likely incorrect")
                        continue
                    
                    logger.info(f"SUCCESS! Captcha accepted on attempt {attempt}")
                    return True
                    
                except TimeoutException:
                    logger.warning(f"Attempt {attempt}: No results found after submission")
                    continue
                    
            except Exception as e:
                logger.error(f"Attempt {attempt}: Error - {str(e)}")
                continue
        
        logger.error("FAILED to solve captcha after all attempts")
        return False

    def extract_table_data(self) -> Tuple[List[List[str]], List[str]]:
        """Extract judgment data from the results table"""
        
        if not self.driver:
            logger.error("Driver not initialized")
            return [], []
        
        time.sleep(2)
        
        try:
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # Find table - try multiple selectors
            table = None
            for selector in ["table", ".dataTables_wrapper table", "#div_datatable table"]:
                table = soup.select_one(selector)
                if table:
                    break
            
            if not table:
                logger.error("No table found in page")
                return [], []
            
            # Extract headers
            header_row = table.find("tr")
            if not header_row:
                logger.error("No header row found")
                return [], []
            
            headers = []
            if header_row and hasattr(header_row, "find_all"):
                for th in header_row.find_all(["th", "td"]) if isinstance(header_row, Tag) else []:
                    text = th.get_text(strip=True)
                    if text:
                        headers.append(text)
            else:
                print("⚠️ header_row is not a valid Tag:", type(header_row))
                if not headers:
                            logger.error("No headers found")
                            return [], []
            
            # Add our custom columns
            headers.extend(["PDF_URL", "PDF_Downloaded", "Local_PDF_Path"])
            
            # Extract data rows
            rows = table.find_all("tr")[1:] if isinstance(table, Tag) else []  # Skip header
            output_data = []
            
            for row_idx, row in enumerate(rows):
                if not isinstance(row, Tag):
                    continue
                    
                cols = row.find_all(["td", "th"]) if isinstance(row, Tag) else []
                row_data = []
                
                for col in cols:
                    text = col.get_text(" ", strip=True) if hasattr(col, 'get_text') else str(col)
                    row_data.append(text)
                
                if not row_data or not any(cell.strip() for cell in row_data):
                    continue
                
                # Look for PDF download link
                pdf_url = ""
                
                # Method 1: Direct link
                link = row.find("a", href=True) if hasattr(row, 'find') else None
                if link and isinstance(link, Tag):
                    href_val = link.get("href")
                    link_text = link.get_text("").lower() if hasattr(link, 'get_text') else ""
                    if href_val and ((".pdf" in str(href_val)) or ("download" in link_text)):
                        pdf_url = urljoin(BASE_URL, str(href_val))
                
                # Method 2: Button with onclick
                if not pdf_url:
                    button = row.find("button") if hasattr(row, 'find') else None
                    if button and isinstance(button, Tag):
                        onclick_val = button.get("onclick")
                        if onclick_val:
                            match = re.search(r"'([^']*\.pdf[^']*)'", str(onclick_val))
                            if match:
                                pdf_url = urljoin(BASE_URL, match.group(1))
                
                # Method 3: Any element with PDF-related onclick
                if not pdf_url:
                    for element in (row.find_all(onclick=True) if isinstance(row, Tag) else []):
                        if isinstance(element, Tag):
                            onclick_text = element.get("onclick")
                            if onclick_text and "pdf" in str(onclick_text).lower():
                                match = re.search(r"'([^']*\.pdf[^']*)'", str(onclick_text))
                                if match:
                                    pdf_url = urljoin(BASE_URL, match.group(1))
                                    break
                
                # Add PDF info to row
                if pdf_url:
                    row_data.extend([pdf_url, "Pending", ""])
                else:
                    row_data.extend(["No PDF Link", "N/A", ""])
                
                # Ensure row has correct number of columns
                while len(row_data) < len(headers):
                    row_data.append("")
                
                output_data.append(row_data)
            
            logger.info(f"Extracted {len(output_data)} rows with {len(headers)} columns")
            return output_data, headers
            
        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return [], []

    def download_pdf(self, pdf_url: str, case_details: List[str], row_identifier: str) -> Tuple[str, str]:
        """Download a single PDF file"""
        if not pdf_url or pdf_url in self.downloaded_pdfs or pdf_url == "No PDF Link":
            return "Skipped", ""
        
        try:
            # Create filename
            case_id = case_details[0] if case_details else f"case_{row_identifier}"
            safe_filename = re.sub(r'[^\w\-_.]', '_', str(case_id))
            pdf_filename = f"{safe_filename}_{row_identifier}.pdf"
            pdf_path = os.path.join(PDF_DIR, pdf_filename)
            
            # Download
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': BASE_URL
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type or response.content.startswith(b'%PDF'):
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                self.downloaded_pdfs.add(pdf_url)
                logger.info(f"Downloaded: {pdf_filename}")
                return "Success", pdf_filename
            else:
                logger.warning(f"Not a PDF file: {pdf_url}")
                return "Failed - Not PDF", ""
                
        except Exception as e:
            logger.error(f"Download failed for {pdf_url}: {e}")
            return f"Failed - {str(e)[:50]}", ""

    def run_scraping_process(self) -> bool:
        """Main function to run the entire scraping process"""
        
        try:
            logger.info("Starting Rajasthan High Court judgment scraper")
            
            # Setup driver
            if not self.setup_driver():
                return False
            
            if not self.driver or not self.wait:
                logger.error("Driver setup failed")
                return False
            
            # Navigate to the page
            self.driver.get(BASE_URL)
            logger.info("Loaded judgment filter page")
            
            # Fill in the form
            logger.info("Filling form with date range and settings")
            
            from_date_input = self.wait.until(EC.presence_of_element_located((By.ID, "partyFromDate")))
            from_date_input.clear()
            from_date_input.send_keys(from_date_str)
            
            to_date_input = self.driver.find_element(By.ID, "partyToDate")
            to_date_input.clear()
            to_date_input.send_keys(to_date_str)
            
            # Select reportable judgments
            reportable_yes = self.driver.find_element(By.ID, "rpjudgeY")
            self.driver.execute_script("arguments[0].click();", reportable_yes)
            
            logger.info(f"Set date range: {from_date_str} to {to_date_str}")
            
            # Solve captcha
            if not self.solve_captcha_with_retry():
                logger.error("Failed to solve captcha, exiting")
                return False
            
            # Extract data
            logger.info("Extracting judgment data from results")
            data_rows, headers = self.extract_table_data()
            
            if not data_rows:
                logger.warning("No data extracted")
                return False
            
            # Store data
            self.all_judgments = [headers] + data_rows
            
            # Download PDFs
            logger.info(f"Starting PDF downloads for {len(data_rows)} judgments")
            
            for row_idx, row_data in enumerate(data_rows):
                pdf_url = row_data[-3]  # PDF_URL column
                
                if pdf_url and pdf_url != "No PDF Link":
                    logger.info(f"Downloading PDF {row_idx+1}/{len(data_rows)}")
                    download_status, local_filename = self.download_pdf(
                        pdf_url, row_data, f"row_{row_idx}"
                    )
                    row_data[-2] = download_status  # Update download status
                    row_data[-1] = local_filename   # Update local path
            
            # Save to CSV
            df = pd.DataFrame(data_rows, columns=headers)
            df.to_csv(CSV_FILE, index=False)
            logger.info(f"Data saved to: {CSV_FILE}")
            
            # Print summary
            total_records = len(data_rows)
            total_pdfs = len(self.downloaded_pdfs)
            
            print(f"\n{'='*50}")
            print(f"SCRAPING COMPLETED SUCCESSFULLY!")
            print(f"{'='*50}")
            print(f"Date range: {from_date_str} to {to_date_str}")
            print(f"Total records: {total_records}")
            print(f"Total PDFs downloaded: {total_pdfs}")
            print(f"CSV file: {CSV_FILE}")
            print(f"PDF folder: {PDF_DIR}")
            print(f"{'='*50}")
            
            return True
            
        except Exception as e:
            logger.error(f"Main process error: {e}")
            return False
            
        finally:
            # Always clean up
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("Browser closed")
                except:
                    pass


def main():
    """Main entry point"""
    scraper = RajasthanHCScraper()
    success = scraper.run_scraping_process()
    
    if success:
        print("\nScript completed successfully!")
        
        # Show some sample data if available
        if os.path.exists(CSV_FILE):
            try:
                df = pd.read_csv(CSV_FILE)
                print(f"\nSample data (first 3 rows):")
                print(df.head(3).to_string(index=False, max_colwidth=40))
            except:
                pass
    else:
        print("\nScript failed. Check the logs above for details.")


if __name__ == "__main__":
    main()