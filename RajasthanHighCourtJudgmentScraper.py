#!/usr/bin/env python3
"""
Fixed Rajasthan High Court Judgment Scraper
Handles the current website structure and fixes captcha detection issues
"""

import os
import csv
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try to import pytesseract with error handling
try:
    import pytesseract
    # For Windows users, uncomment and set the correct path:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Warning: Tesseract not available. Captcha solving will be manual.")
    TESSERACT_AVAILABLE = False

class FixedRajasthanHCScraper:
    def __init__(self, download_dir: str = "rajasthan_hc_judgments"):
        self.base_url = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/"
        self.download_dir = Path(download_dir)
        self.pdf_dir = self.download_dir / "pdfs"
        self.csv_file = self.download_dir / "judgments.csv"
        self.state_file = self.download_dir / "scraper_state.json"
        
        # Create directories
        self.download_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.downloaded_judgments = self.load_state()
        
        # Setup Chrome options
        self.chrome_options = Options()
        # Comment out headless mode for debugging
        # self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Set download preferences
        prefs = {
            "download.default_directory": str(self.pdf_dir.absolute()),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_settings.popups": 0
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
        
    def load_state(self) -> Dict:
        """Load previously downloaded judgment IDs and metadata"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    state["downloaded_ids"] = set(state.get("downloaded_ids", []))
                    return state
            except Exception as e:
                print(f"Error loading state: {e}")
        return {"downloaded_ids": set(), "last_run_date": None}
    
    def save_state(self):
        """Save current state to file"""
        state_to_save = {
            "downloaded_ids": list(self.downloaded_judgments["downloaded_ids"]),
            "last_run_date": self.downloaded_judgments["last_run_date"]
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2)
    
    def generate_judgment_id(self, judgment_data: Dict) -> str:
        """Generate unique ID for judgment based on key fields"""
        # Use multiple fields to create a more unique ID
        id_parts = []
        for key, value in list(judgment_data.items())[:5]:  # First 5 fields
            if value and str(value).strip():
                id_parts.append(str(value).strip())
        
        id_string = "|".join(id_parts) if id_parts else str(hash(str(judgment_data)))
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def enhance_captcha_image(self, image_path: str) -> str:
        """Enhanced captcha image processing"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Increase image size for better OCR
            height, width = gray.shape
            if height < 50:
                scale_factor = 3
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply multiple preprocessing techniques
            
            # Method 1: Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Median filter + adaptive threshold
            denoised = cv2.medianBlur(gray, 3)
            thresh2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Method 3: Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
            
            # Try OCR on all processed images
            processed_images = [thresh1, thresh2, thresh3, gray]
            
            for i, processed_img in enumerate(processed_images):
                try:
                    # Different OCR configurations
                    configs = [
                        r'--oem 3 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        r'--oem 3 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        r'--oem 3 --psm 6 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        r'--oem 3 --psm 13'
                    ]
                    
                    for config in configs:
                        text = pytesseract.image_to_string(processed_img, config=config).strip()
                        # Clean the text
                        cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                        
                        if len(cleaned_text) >= 4 and len(cleaned_text) <= 8:
                            print(f"OCR success with method {i+1}, config {config[:20]}...: {cleaned_text}")
                            return cleaned_text
                
                except Exception as e:
                    continue
            
            return ""
            
        except Exception as e:
            print(f"Error in captcha enhancement: {e}")
            return ""
    
    def solve_captcha_ocr(self, driver) -> str:
        """Advanced OCR captcha solving"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            print("Looking for captcha image...")
            
            # Multiple ways to find captcha image
            captcha_selectors = [
                "//img[contains(@src, 'captcha')]",
                "//img[contains(@src, 'Captcha')]",
                "//img[contains(@src, 'CAPTCHA')]",
                "//img[contains(@id, 'captcha')]",
                "//img[contains(@id, 'Captcha')]",
                "//img[contains(@class, 'captcha')]",
                "//img[contains(@alt, 'captcha')]",
                "//img[contains(@title, 'captcha')]"
            ]
            
            captcha_img = None
            for selector in captcha_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        captcha_img = elements[0]
                        print(f"Found captcha with selector: {selector}")
                        break
                except:
                    continue
            
            if not captcha_img:
                print("No captcha image found with standard selectors")
                # Try to find any image near captcha input
                try:
                    captcha_input = driver.find_element(By.ID, "txtCaptcha")
                    # Look for images near the captcha input
                    nearby_imgs = driver.find_elements(By.XPATH, "//img")
                    if nearby_imgs:
                        captcha_img = nearby_imgs[-1]  # Often the last image is captcha
                        print("Using last image as potential captcha")
                except:
                    pass
            
            if not captcha_img:
                print("Could not locate captcha image")
                return ""
            
            # Take screenshot of captcha
            captcha_img.screenshot("captcha_temp.png")
            print("Captcha image saved")
            
            # Enhanced processing
            captcha_text = self.enhance_captcha_image("captcha_temp.png")
            
            # Clean up
            if os.path.exists("captcha_temp.png"):
                os.remove("captcha_temp.png")
            
            return captcha_text
            
        except Exception as e:
            print(f"Error in OCR captcha solving: {e}")
            return ""
    
    def solve_captcha_manual(self, driver) -> str:
        """Manual captcha input with better image display"""
        try:
            print("Attempting manual captcha input...")
            
            # Try to save captcha image for manual review
            captcha_saved = False
            try:
                # Look for captcha image
                captcha_selectors = [
                    "//img[contains(@src, 'captcha')]",
                    "//img[contains(@src, 'Captcha')]",
                    "//img[contains(@id, 'captcha')]"
                ]
                
                for selector in captcha_selectors:
                    try:
                        captcha_imgs = driver.find_elements(By.XPATH, selector)
                        if captcha_imgs:
                            captcha_imgs[0].screenshot("manual_captcha.png")
                            print("Captcha image saved as 'manual_captcha.png'")
                            captcha_saved = True
                            break
                    except:
                        continue
                
                if not captcha_saved:
                    # Try alternative method
                    all_imgs = driver.find_elements(By.TAG_NAME, "img")
                    for img in reversed(all_imgs):  # Start from last image
                        try:
                            src = img.get_attribute("src") or ""
                            if "captcha" in src.lower() or img.size['height'] < 100:
                                img.screenshot("manual_captcha.png")
                                print("Captcha image saved as 'manual_captcha.png' (alternative method)")
                                captcha_saved = True
                                break
                        except:
                            continue
                            
            except Exception as e:
                print(f"Could not save captcha image: {e}")
            
            if captcha_saved:
                print("\nPlease check 'manual_captcha.png' file to see the captcha.")
            
            print("If running in Jupyter, you can display the image with:")
            print("from IPython.display import Image, display")
            print("display(Image('manual_captcha.png'))")
            
            captcha_text = input("\nEnter the captcha text (or press Enter to skip): ").strip()
            return captcha_text.upper()
            
        except KeyboardInterrupt:
            print("\nCaptcha input cancelled by user")
            return ""
        except Exception as e:
            print(f"Error in manual captcha: {e}")
            return ""
    
    def solve_captcha(self, driver) -> str:
        """Comprehensive captcha solving"""
        print("Solving captcha...")
        
        # First try OCR
        if TESSERACT_AVAILABLE:
            ocr_result = self.solve_captcha_ocr(driver)
            if ocr_result:
                return ocr_result
        
        # Fallback to manual input
        return self.solve_captcha_manual(driver)
    
    def setup_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver"""
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            raise
    
    def wait_and_click(self, driver, element, max_attempts=3):
        """Reliable click with multiple attempts"""
        for attempt in range(max_attempts):
            try:
                # Scroll to element
                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.5)
                
                # Try different click methods
                try:
                    element.click()
                    return True
                except:
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        return True
                    except:
                        ActionChains(driver).move_to_element(element).click().perform()
                        return True
                        
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Failed to click element after {max_attempts} attempts: {e}")
                    return False
                time.sleep(1)
        return False
    
    def navigate_to_search_form(self, driver: webdriver.Chrome) -> bool:
        """Navigate to the search form"""
        try:
            print("Loading main page...")
            driver.get(self.base_url)
            
            # Wait for page load
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)
            
            print("Looking for Multiple Parameter Search...")
            
            # Try to find the Multiple Parameter Search button
            search_selectors = [
                "//a[contains(text(), 'Multiple Parameter Search')]",
                "//button[contains(text(), 'Multiple Parameter Search')]",
                "//input[contains(@value, 'Multiple Parameter Search')]",
                "//div[contains(text(), 'Multiple Parameter Search')]//ancestor-or-self::*[@onclick or @href]",
                "//*[contains(text(), 'Multiple Parameter Search') and (@onclick or @href or self::button or self::a)]"
            ]
            
            search_button = None
            for selector in search_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        search_button = elements[0]
                        print(f"Found search button with: {selector[:50]}...")
                        break
                except:
                    continue
            
            if not search_button:
                print("Could not find Multiple Parameter Search button")
                print("Available clickable elements:")
                clickables = driver.find_elements(By.XPATH, "//a | //button | //input[@type='button'] | //input[@type='submit']")
                for i, elem in enumerate(clickables[:10]):
                    try:
                        text = (elem.text or elem.get_attribute('value') or elem.get_attribute('title') or "")[:50]
                        if text:
                            print(f"  {i+1}. {text}")
                    except:
                        continue
                return False
            
            # Click the search button
            if self.wait_and_click(driver, search_button):
                print("Successfully clicked Multiple Parameter Search")
                time.sleep(3)
                
                # Wait for the form to load
                try:
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "form")))
                    print("Search form loaded successfully")
                    return True
                except:
                    print("Form did not load properly")
                    return False
            else:
                print("Failed to click Multiple Parameter Search button")
                return False
                
        except Exception as e:
            print(f"Error navigating to search form: {e}")
            return False
    
    def fill_search_form(self, driver: webdriver.Chrome, from_date: str, to_date: str, max_retries: int = 3) -> bool:
        """Fill and submit the search form"""
        try:
            print(f"Filling search form: {from_date} to {to_date}")
            
            wait = WebDriverWait(driver, 10)
            
            # Find date fields
            from_date_field = None
            to_date_field = None
            
            # Try multiple ways to find date fields
            date_selectors = [
                ("partyFromDate", "partyToDate"),  # Based on your log output
                ("fromDate", "toDate"),
                ("from_date", "to_date"),
                ("dateFrom", "dateTo")
            ]
            
            for from_id, to_id in date_selectors:
                try:
                    from_date_field = driver.find_element(By.ID, from_id)
                    to_date_field = driver.find_element(By.ID, to_id)
                    print(f"Found date fields: {from_id}, {to_id}")
                    break
                except:
                    continue
            
            if not from_date_field or not to_date_field:
                print("Could not find date fields")
                return False
            
            # Clear and fill date fields
            print("Filling date fields...")
            from_date_field.clear()
            from_date_field.send_keys(from_date)
            
            to_date_field.clear()
            to_date_field.send_keys(to_date)
            
            # Set Reportable Judgment to YES
            try:
                # Look for reportable judgment radio buttons
                reportable_yes_selectors = [
                    "//input[@id='rpjudgeY']",
                    "//input[@name='rpjudge' and @value='Y']",
                    "//input[contains(@id, 'reportable') and @value='Y']"
                ]
                
                for selector in reportable_yes_selectors:
                    try:
                        reportable_yes = driver.find_element(By.XPATH, selector)
                        reportable_yes.click()
                        print("Set Reportable Judgment to YES")
                        break
                    except:
                        continue
                        
            except Exception as e:
                print(f"Could not set Reportable Judgment: {e}")
            
            # Handle captcha and submit
            for attempt in range(max_retries):
                try:
                    print(f"\nCaptcha attempt {attempt + 1}/{max_retries}")
                    
                    # Get captcha text
                    captcha_text = self.solve_captcha(driver)
                    
                    if not captcha_text:
                        print("No captcha text provided")
                        if attempt < max_retries - 1:
                            print("Refreshing page...")
                            driver.refresh()
                            time.sleep(3)
                            if not self.navigate_to_search_form(driver):
                                return False
                            continue
                        else:
                            return False
                    
                    # Find captcha input field
                    captcha_field = None
                    captcha_selectors = ["txtCaptcha", "captcha", "Captcha"]
                    
                    for selector in captcha_selectors:
                        try:
                            captcha_field = driver.find_element(By.ID, selector)
                            break
                        except:
                            continue
                    
                    if not captcha_field:
                        try:
                            captcha_field = driver.find_element(By.XPATH, "//input[contains(@name, 'captcha') or contains(@name, 'Captcha')]")
                        except:
                            print("Could not find captcha input field")
                            return False
                    
                    # Enter captcha
                    captcha_field.clear()
                    captcha_field.send_keys(captcha_text)
                    print(f"Entered captcha: {captcha_text}")
                    
                    # Find and click submit button
                    submit_selectors = [
                        "//input[@type='submit']",
                        "//button[@type='submit']",
                        "//input[contains(@value, 'Search')]",
                        "//button[contains(text(), 'Search')]",
                        "//input[contains(@value, 'Submit')]"
                    ]
                    
                    submit_button = None
                    for selector in submit_selectors:
                        try:
                            buttons = driver.find_elements(By.XPATH, selector)
                            if buttons:
                                submit_button = buttons[0]
                                break
                        except:
                            continue
                    
                    if not submit_button:
                        print("Could not find submit button")
                        return False
                    
                    # Submit the form
                    if self.wait_and_click(driver, submit_button):
                        print("Form submitted")
                        time.sleep(5)  # Wait for results
                        
                        # Check if submission was successful
                        page_source = driver.page_source.lower()
                        current_url = driver.current_url.lower()
                        
                        # Check for error indicators
                        if any(error in page_source for error in ["invalid captcha", "incorrect captcha", "wrong captcha"]):
                            print("Invalid captcha - retrying...")
                            continue
                        elif "no records found" in page_source:
                            print("No records found for the given date range")
                            return True  # This is a valid response, just no data
                        elif any(success in page_source for success in ["judgment", "case", "result"]) or "result" in current_url:
                            print("Form submitted successfully!")
                            return True
                        else:
                            print("Unclear submission result - checking page content...")
                            # If we see a table or list, consider it successful
                            if driver.find_elements(By.TAG_NAME, "table") or driver.find_elements(By.XPATH, "//tr"):
                                print("Found table data - considering successful")
                                return True
                            else:
                                print("No clear success indicator - retrying...")
                                continue
                    else:
                        print("Failed to click submit button")
                        
                except Exception as e:
                    print(f"Error in attempt {attempt + 1}: {e}")
                
                # Refresh for retry
                if attempt < max_retries - 1:
                    print("Refreshing page for retry...")
                    driver.refresh()
                    time.sleep(3)
                    if not self.navigate_to_search_form(driver):
                        return False
                    if not self.fill_search_form(driver, from_date, to_date, 1):  # Single retry
                        continue
            
            print("All submission attempts failed")
            return False
            
        except Exception as e:
            print(f"Error filling search form: {e}")
            return False
    
    def extract_judgment_data(self, driver: webdriver.Chrome) -> List[Dict]:
        """Extract judgment data from results"""
        judgments = []
        
        try:
            print("Extracting judgment data...")
            
            # Wait a bit for results to load
            time.sleep(3)
            
            # Look for results table
            tables = driver.find_elements(By.TAG_NAME, "table")
            print(f"Found {len(tables)} tables on page")
            
            results_table = None
            for i, table in enumerate(tables):
                table_text = table.text.lower()
                print(f"Table {i+1} preview: {table_text[:100]}...")
                
                # Look for table with judgment data
                if any(keyword in table_text for keyword in ["s.no", "case", "judgment", "date", "party", "serial"]):
                    results_table = table
                    print(f"Selected table {i+1} as results table")
                    break
            
            if not results_table:
                print("No results table found")
                # Try to find any structured data
                rows = driver.find_elements(By.XPATH, "//tr[position()>1 and count(td)>2]")
                if rows:
                    print(f"Found {len(rows)} data rows without clear table structure")
                    # Process rows without table structure
                    for i, row in enumerate(rows):
                        try:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) >= 3:  # At least 3 columns
                                judgment_data = {}
                                for j, cell in enumerate(cells):
                                    judgment_data[f"Column_{j+1}"] = cell.text.strip()
                                
                                # Look for PDF links
                                pdf_links = row.find_elements(By.XPATH, ".//a[contains(@href, '.pdf') or contains(text(), 'View') or contains(text(), 'Download')]")
                                if pdf_links:
                                    judgment_data['pdf_url'] = pdf_links[0].get_attribute('href')
                                else:
                                    judgment_data['pdf_url'] = ""
                                
                                judgments.append(judgment_data)
                        except Exception as e:
                            print(f"Error processing row {i}: {e}")
                            continue
                
                return judgments
            
            # Extract headers
            headers = []
            try:
                header_row = results_table.find_element(By.TAG_NAME, "tr")
                header_cells = header_row.find_elements(By.TAG_NAME, "th")
                if not header_cells:
                    header_cells = header_row.find_elements(By.TAG_NAME, "td")
                
                for cell in header_cells:
                    header_text = cell.text.strip()
                    if not header_text:
                        header_text = f"Column_{len(headers)+1}"
                    headers.append(header_text)
                    
            except:
                headers = []
            
            if not headers:
                # Use default headers
                first_row = results_table.find_elements(By.TAG_NAME, "tr")[0] if results_table.find_elements(By.TAG_NAME, "tr") else None
                if first_row:
                    cell_count = len(first_row.find_elements(By.TAG_NAME, "td"))
                    headers = [f"Column_{i+1}" for i in range(cell_count)]
            
            print(f"Table headers: {headers}")
            
            # Extract data rows
            rows = results_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
            print(f"Processing {len(rows)} data rows...")
            
            for i, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if not cells:
                        continue
                    
                    judgment_data = {}
                    for j, cell in enumerate(cells):
                        header_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                        cell_text = cell.text.strip()
                        judgment_data[header_name] = cell_text
                    
                    # Look for PDF download links
                    pdf_links = row.find_elements(By.XPATH, ".//a[contains(@href, '.pdf') or contains(@href, 'download') or contains(text(), 'View') or contains(text(), 'Download')]")
                    if pdf_links:
                        judgment_data['pdf_url'] = pdf_links[0].get_attribute('href')
                        print(f"Found PDF link for row {i+1}")
                    else:
                        judgment_data['pdf_url'] = ""
                    
                    # Only add if we have meaningful data
                    if any(value and str(value).strip() for value in judgment_data.values() if value != judgment_data.get('pdf_url', '')):
                        judgments.append(judgment_data)
                        
                except Exception as e:
                    print(f"Error processing row {i+1}: {e}")
                    continue
            
            print(f"Successfully extracted {len(judgments)} judgment records")
            
            # Show sample of extracted data
            if judgments:
                print("Sample judgment data:")
                sample = judgments[0]
                for key, value in sample.items():
                    if value:
                        print(f"  {key}: {str(value)[:50]}...")
            
            return judgments
            
        except Exception as e:
            print(f"Error extracting judgment data: {e}")
            return judgments
    
    def download_pdf(self, url: str, filename: str) -> bool:
        """Download PDF from URL"""
        try:
            print(f"Downloading: {filename}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*',
                'Referer': 'https://hcraj.nic.in/'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Ensure it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                print(f"Warning: URL may not be a PDF: {content_type}")
            
            pdf_path = self.pdf_dir / filename
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = pdf_path.stat().st_size
            if file_size < 1024:  # Less than 1KB, likely an error page
                print(f"Downloaded file is too small ({file_size} bytes), likely an error")
                return False
            
            print(f"Successfully downloaded {filename} ({file_size} bytes)")
            return True
            
        except Exception as e:
            print(f"Error downloading PDF {filename}: {e}")
            return False
    
    def generate_pdf_filename(self, judgment_data: Dict) -> str:
        """Generate safe filename for PDF"""
        # Try to find case number and date
        case_num = "Unknown"
        date_str = "Unknown"
        
        for key, value in judgment_data.items():
            if value and str(value).strip():
                key_lower = key.lower()
                value_str = str(value).strip()
                
                if any(keyword in key_lower for keyword in ["case", "number", "no"]):
                    case_num = value_str.replace('/', '_').replace('\\', '_')[:50]
                elif any(keyword in key_lower for keyword in ["date", "judgment"]):
                    date_str = value_str.replace('/', '_').replace('-', '_')[:20]
        
        # Create safe filename
        safe_filename = f"{case_num}_{date_str}.pdf"
        # Remove invalid characters
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
        
        # Ensure it doesn't start with dot or dash
        if safe_filename.startswith(('.', '-')):
            safe_filename = 'judgment_' + safe_filename
        
        return safe_filename[:100]  # Limit length
    
    def scrape_judgments(self, from_date: str, to_date: str) -> List[Dict]:
        """Main scraping function"""
        print(f"Scraping judgments from {from_date} to {to_date}")
        
        try:
            driver = self.setup_driver()
        except Exception as e:
            print(f"Failed to setup driver: {e}")
            return []
        
        all_judgments = []
        
        try:
            # Navigate to search form
            if not self.navigate_to_search_form(driver):
                print("Could not navigate to search form")
                return []
            
            # Fill and submit form
            if not self.fill_search_form(driver, from_date, to_date):
                print("Could not fill and submit form")
                return []
            
            # Extract results
            judgments = self.extract_judgment_data(driver)
            
            if not judgments:
                print("No judgments found")
                return []
            
            print(f"Found {len(judgments)} judgments to process")
            
            # Process each judgment
            for i, judgment in enumerate(judgments):
                try:
                    judgment_id = self.generate_judgment_id(judgment)
                    
                    # Skip if already downloaded
                    if judgment_id in self.downloaded_judgments["downloaded_ids"]:
                        case_ref = list(judgment.values())[0] if judgment else "Unknown"
                        print(f"Skipping already downloaded: {case_ref[:50]}...")
                        continue
                    
                    # Download PDF if URL exists
                    pdf_filename = ""
                    if judgment.get('pdf_url') and judgment['pdf_url'].strip():
                        pdf_filename = self.generate_pdf_filename(judgment)
                        
                        # Ensure unique filename
                        counter = 1
                        original_filename = pdf_filename
                        while (self.pdf_dir / pdf_filename).exists():
                            name, ext = os.path.splitext(original_filename)
                            pdf_filename = f"{name}_{counter}{ext}"
                            counter += 1
                        
                        if self.download_pdf(judgment['pdf_url'], pdf_filename):
                            judgment['pdf_filename'] = pdf_filename
                            judgment['download_status'] = "Success"
                        else:
                            judgment['pdf_filename'] = "Download_Failed"
                            judgment['download_status'] = "Failed"
                    else:
                        judgment['pdf_filename'] = "No_PDF_URL"
                        judgment['download_status'] = "No_URL"
                    
                    # Add metadata
                    judgment['scraped_date'] = datetime.now().isoformat()
                    judgment['judgment_id'] = judgment_id
                    
                    # Mark as processed
                    self.downloaded_judgments["downloaded_ids"].add(judgment_id)
                    all_judgments.append(judgment)
                    
                    print(f"Processed judgment {i+1}/{len(judgments)}: {judgment.get('pdf_filename', 'No PDF')}")
                    
                except Exception as e:
                    print(f"Error processing judgment {i+1}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error during scraping: {e}")
        
        finally:
            try:
                driver.quit()
            except:
                pass
        
        return all_judgments
    
    def save_to_csv(self, judgments: List[Dict]):
        """Save judgments to CSV file"""
        if not judgments:
            print("No new judgments to save")
            return
        
        # Load existing data
        existing_df = pd.DataFrame()
        if self.csv_file.exists():
            try:
                existing_df = pd.read_csv(self.csv_file)
                print(f"Loaded {len(existing_df)} existing records")
            except Exception as e:
                print(f"Warning: Could not load existing CSV: {e}")
        
        # Create new DataFrame
        new_df = pd.DataFrame(judgments)
        
        # Combine data
        if not existing_df.empty:
            # Ensure columns match
            all_columns = set(existing_df.columns) | set(new_df.columns)
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = ""
                if col not in new_df.columns:
                    new_df[col] = ""
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates based on judgment_id if present
        if 'judgment_id' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['judgment_id'], keep='last')
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                print(f"Removed {before_dedup - after_dedup} duplicate records")
        
        # Save to CSV
        combined_df.to_csv(self.csv_file, index=False)
        print(f"Saved {len(judgments)} new judgments to {self.csv_file}")
        print(f"Total judgments in database: {len(combined_df)}")
    
    def run_incremental_scrape(self, days_back: int = 10):
        """Run incremental scraping"""
        today = datetime.now()
        from_date_obj = today - timedelta(days=days_back)
        
        from_date = from_date_obj.strftime("%d/%m/%Y")
        to_date = today.strftime("%d/%m/%Y")
        
        print(f"Running incremental scrape from {from_date} to {to_date}")
        print(f"Looking for judgments in the last {days_back} days")
        
        judgments = self.scrape_judgments(from_date, to_date)
        
        if judgments:
            self.save_to_csv(judgments)
        
        # Update state
        self.downloaded_judgments["last_run_date"] = today.isoformat()
        self.save_state()
        
        print(f"\nScraping completed!")
        print(f"- Downloaded {len(judgments)} new judgments")
        print(f"- Files saved in: {self.download_dir}")
        print(f"- CSV file: {self.csv_file}")
        print(f"- PDFs saved in: {self.pdf_dir}")
        
        return judgments

# Bonus: Supreme Court of India Captcha Solver
class SCICaptchaSolver:
    def __init__(self):
        """Initialize SCI captcha solver with offline ML models"""
        self.models_initialized = False
        self.setup_models()
    
    def setup_models(self):
        """Setup offline ML models for captcha solving"""
        try:
            # Use traditional CV + OCR approach (offline)
            if TESSERACT_AVAILABLE:
                print("SCI Captcha Solver initialized with OCR")
                self.models_initialized = True
            else:
                print("Tesseract not available for SCI captcha solving")
        except Exception as e:
            print(f"Error setting up SCI captcha models: {e}")
    
    def preprocess_sci_captcha(self, image_path: str) -> List[np.ndarray]:
        """Advanced preprocessing for SCI captcha images"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize for better processing
            height, width = gray.shape
            if height < 60:
                scale = 60 / height
                new_width = int(width * scale)
                gray = cv2.resize(gray, (new_width, 60), interpolation=cv2.INTER_CUBIC)
            
            processed_images = []
            
            # Method 1: Gaussian blur + Otsu threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh1)
            
            # Method 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_images.append(adaptive)
            
            # Method 3: Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            processed_images.append(morph)
            
            # Method 4: Edge detection + dilation
            edges = cv2.Canny(gray, 50, 150)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            processed_images.append(dilated)
            
            # Method 5: Bilateral filter + threshold
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh2)
            
            return processed_images
            
        except Exception as e:
            print(f"Error preprocessing SCI captcha: {e}")
            return []
    
    def solve_sci_captcha(self, captcha_image_path: str) -> str:
        """Solve Supreme Court captcha using offline methods"""
        if not self.models_initialized:
            return ""
        
        try:
            print(f"Solving SCI captcha: {captcha_image_path}")
            
            # Preprocess image with multiple methods
            processed_images = self.preprocess_sci_captcha(captcha_image_path)
            
            if not processed_images:
                return ""
            
            # Try OCR on each processed image
            for i, processed_img in enumerate(processed_images):
                try:
                    # Different OCR configurations for SCI captcha
                    configs = [
                        r'--oem 3 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        r'--oem 3 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        r'--oem 3 --psm 6',
                        r'--oem 3 --psm 13'
                    ]
                    
                    for config in configs:
                        try:
                            text = pytesseract.image_to_string(processed_img, config=config).strip()
                            # Clean and validate
                            cleaned = ''.join(c for c in text if c.isalnum()).upper()
                            
                            # SCI captchas are usually 5-6 characters
                            if 4 <= len(cleaned) <= 8 and cleaned.isalnum():
                                print(f"SCI captcha solved: {cleaned} (method {i+1})")
                                return cleaned
                                
                        except Exception as e:
                            continue
                
                except Exception as e:
                    continue
            
            print("Could not solve SCI captcha with OCR")
            return ""
            
        except Exception as e:
            print(f"Error solving SCI captcha: {e}")
            return ""
    
    def test_sci_solver(self, test_image_path: str):
        """Test the SCI captcha solver"""
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            return
        
        result = self.solve_sci_captcha(test_image_path)
        print(f"Test result for {test_image_path}: {result}")

def show_results(scraper):
    """Display scraping results"""
    if scraper.csv_file.exists():
        try:
            df = pd.read_csv(scraper.csv_file)
            print(f"\n--- SCRAPING RESULTS ---")
            print(f"Total judgments: {len(df)}")
            
            if len(df) > 0:
                print("\nColumn names:")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i}. {col}")
                
                print("\nDownload statistics:")
                if 'download_status' in df.columns:
                    status_counts = df['download_status'].value_counts()
                    for status, count in status_counts.items():
                        print(f"  {status}: {count}")
                
                print("\nSample data:")
                print(df.head(2).to_string(max_colwidth=50))
                
        except Exception as e:
            print(f"Error reading results: {e}")
    else:
        print("No results file found")

def main():
    """Main function"""
    print("FIXED RAJASTHAN HIGH COURT JUDGMENT SCRAPER")
    print("=" * 60)
    
    try:
        # Initialize scraper
        scraper = FixedRajasthanHCScraper()
        
        # Run scraping
        judgments = scraper.run_incremental_scrape()
        
        # Show results
        show_results(scraper)
        
        # Initialize bonus SCI captcha solver
        print("\n" + "=" * 60)
        print("BONUS: Supreme Court Captcha Solver")
        print("=" * 60)
        
        sci_solver = SCICaptchaSolver()
        print("SCI Captcha solver ready. Usage:")
        print("sci_solver.solve_sci_captcha('path_to_captcha.png')")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
                