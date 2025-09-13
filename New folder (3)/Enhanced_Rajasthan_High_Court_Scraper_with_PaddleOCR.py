#!/usr/bin/env python3
"""
Enhanced Rajasthan High Court Judgment Scraper
Using PaddleOCR for better captcha recognition
"""

import os
import json
import time
import hashlib
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import warnings
import cv2
import numpy as np
import re
import base64
from io import BytesIO
from typing import Any, Optional, Dict, Set
import base64
from urllib.parse import urljoin
from selenium.webdriver.common.by import By
import cv2
import numpy as np
import requests
import re
from typing import Optional, Any
from automatic_captcha_solver import integrate_automatic_captcha_solver
from numerical_captcha_extractor import NumericalCaptchaExtractor, numerical_captcha_extractor
warnings.filterwarnings('ignore')

# # Try to import PaddleOCR
# try:
#     from paddleocr import PaddleOCR
#     PADDLEOCR_AVAILABLE = True
#     print("✅ PaddleOCR available for automatic captcha solving")
# except ImportError:
#     PADDLEOCR_AVAILABLE = False
#     print("⚠️  PaddleOCR not available - install with: pip install paddleocr")
# Safe import pattern for PaddleOCR (prevents "possibly unbound" warnings)
PaddleOCR = None  # type: Any  # define name so static analyzer sees it
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLEOCR_AVAILABLE = True
except Exception:
    PaddleOCR = None  # type: ignore
    PADDLEOCR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rajasthan_scraper.log"),
        logging.StreamHandler()
    ]
)

def get_india_time():
    """Get current time in India"""
    try:
        response = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return datetime.fromisoformat(data["datetime"])
    except:
        pass
    return datetime.now()

class RajasthanHCScraper:
    def __init__(self, download_dir: str = "rajasthan_judgments", headless: bool = False):
        self.base_url = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/"
        self.download_dir = Path(download_dir)
        self.pdf_dir = self.download_dir / "pdfs"
        self.csv_file = self.download_dir / "judgments.csv"
        self.state_file = self.download_dir / "scraper_state.json"

        # Create directories (use parents=True to be robust)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Logger and HTTP session (logger available for load_state)
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()

        # Captcha extractor instance
        self.numerical_captcha_extractor = NumericalCaptchaExtractor()

        # Integrate automatic captcha solver
        scraper = RajasthanHCScraper(headless=False)
        integrate_automatic_captcha_solver(scraper)

        # Setup Chrome options
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")

        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--window-size=1366,768")

        # Download preferences
        prefs = {
            "download.default_directory": str(self.pdf_dir.absolute()),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
        }
        self.chrome_options.add_experimental_option("prefs", prefs)

        # Load state (logger already initialized)
        # Load state (allow None/sets etc. in the state dict)
        self.state: Dict[str, Any] = self.load_state()


        # Initialize PaddleOCR client (if available)
        self.ocr: Optional[Any] = None
        if PADDLEOCR_AVAILABLE and PaddleOCR is not None:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception as e:
                # If initialization fails, keep self.ocr as None but log the error
                self.logger.debug(f"PaddleOCR init failed: {e}")
                self.ocr = None


    def load_state(self) -> Dict[str, Any]:
        """Load scraper state from JSON and normalize values.

        Returns a dict with:
        - 'downloaded_ids' -> set[str]
        - 'last_run' -> optional str (or None)
        """
        default_state: Dict[str, Any] = {"downloaded_ids": set(), "last_run": None}

        try:
            if not self.state_file.exists():
                return default_state

            with self.state_file.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}

            # Normalize downloaded_ids (JSON cannot store a set, so accept list/tuple/str)
            raw_ids = raw.get("downloaded_ids", [])
            downloaded_ids: Set[str]
            if isinstance(raw_ids, list) or isinstance(raw_ids, tuple):
                downloaded_ids = {str(x) for x in raw_ids if x is not None}
            elif isinstance(raw_ids, set):
                downloaded_ids = {str(x) for x in raw_ids}
            elif isinstance(raw_ids, str):
                # single id stored as string
                downloaded_ids = {raw_ids}
            else:
                downloaded_ids = set()

            last_run = raw.get("last_run")
            return {"downloaded_ids": downloaded_ids, "last_run": last_run}

        except Exception as e:
            # Use logger if available; fall back to print if not
            try:
                self.logger.debug(f"Failed to load state from {self.state_file}: {e}")
            except Exception:
                print(f"Failed to load state from {self.state_file}: {e}")
            return default_state

    def save_state(self):
            """Save scraper state"""
            state_to_save = {
                "downloaded_ids": list(self.state["downloaded_ids"]),
                "last_run": datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2)
        
    def generate_judgment_id(self, judgment_data):
            """Generate unique ID for judgment"""
            key_parts = []
            for key, value in judgment_data.items():
                if value and str(value).strip():
                    key_parts.append(str(value).strip())
                    if len(key_parts) >= 3:  # Use first 3 non-empty fields
                        break
            
            id_string = "|".join(key_parts) if key_parts else str(time.time())
            return hashlib.md5(id_string.encode()).hexdigest()
        
    def extract_captcha_from_base64(self, driver) -> str:
        """Enhanced captcha extraction for 6-digit numerical captchas"""
        if not hasattr(self, 'numerical_captcha_extractor'):
            self.numerical_captcha_extractor = NumericalCaptchaExtractor()
        
        if not self.numerical_captcha_extractor.ocr:
            return ""
        
        try:
            captcha_elem = driver.find_element(By.ID, "captcha")
            src = captcha_elem.get_attribute("src")
            if not src:
                return ""

            if src.startswith("data:"):
                result = self.numerical_captcha_extractor.extract_from_base64(src)
            else:
                import requests
                from urllib.parse import urljoin
                
                sess = requests.Session()
                for c in driver.get_cookies():
                    try:
                        sess.cookies.set(c.get("name"), c.get("value"), domain=c.get("domain"))
                    except Exception:
                        sess.cookies.set(c.get("name"), c.get("value"))
                
                img_url = urljoin(self.base_url, src)
                resp = sess.get(img_url, timeout=10)
                resp.raise_for_status()
                
                result = self.numerical_captcha_extractor.extract_from_base64(
                    base64.b64encode(resp.content).decode()
                )
            
            return result

        except Exception as e:
            try:
                self.logger.debug(f"Enhanced captcha extraction error: {e}")
            except Exception:
                print(f"Enhanced captcha extraction error: {e}")
            return ""

    def solve_captcha_manual(self, driver):
            """Manual captcha solving"""
            try:
                print("\n" + "="*50)
                print("MANUAL CAPTCHA INPUT REQUIRED")
                print("="*50)
                
                # Try to save captcha for viewing
                try:
                    captcha_img = driver.find_element(By.ID, "captcha")
                    src_data = captcha_img.get_attribute("src")
                    
                    if src_data and src_data.startswith("data:image"):
                        # Save base64 image
                        base64_data = src_data.split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        
                        with open("captcha_manual.png", "wb") as f:
                            f.write(image_data)
                        print("✅ Captcha saved as 'captcha_manual.png'")
                        print("Please open this file to view the captcha.")
                    else:
                        # Fallback: screenshot the element
                        captcha_img.screenshot("captcha_manual.png")
                        print("✅ Captcha screenshot saved as 'captcha_manual.png'")
                except Exception as e:
                    print(f"Could not save captcha image: {e}")
                
                captcha_text = input("\nEnter captcha text (or press Enter to skip): ").strip()
                return captcha_text.upper() if captcha_text else ""
                
            except KeyboardInterrupt:
                return ""
        
    def setup_driver(self):
            """Setup Chrome driver"""
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.set_page_load_timeout(30)
            return driver
        
    def fill_form_and_search(self, driver, from_date, to_date):
            """Fill the search form and submit with enhanced captcha handling"""
            try:
                wait = WebDriverWait(driver, 15)

                # Helper to find element by possible IDs/names
                def find_input_by_candidates(candidates):
                    for sel in candidates:
                        try:
                            el = driver.find_element(By.ID, sel)
                            return el
                        except:
                            try:
                                el = driver.find_element(By.NAME, sel)
                                return el
                            except:
                                continue
                    return None

                # 1) Set date fields
                from_candidates = ['partyFromDate', 'partyFromdate', 'partyfromdate', 'txtFromDate', 'fromDate', 'from_date']
                to_candidates = ['partyToDate', 'partyTodate', 'partytodate', 'txtToDate', 'toDate', 'to_date']

                from_el = find_input_by_candidates(from_candidates)
                to_el = find_input_by_candidates(to_candidates)

                if from_el:
                    driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input'));", from_el, from_date)
                else:
                    for sel in from_candidates:
                        script = f"var e=document.getElementById('{sel}'); if(e){{ e.value = '{from_date}'; e.dispatchEvent(new Event('input')); }}"
                        driver.execute_script(script)

                if to_el:
                    driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input'));", to_el, to_date)
                else:
                    for sel in to_candidates:
                        script = f"var e=document.getElementById('{sel}'); if(e){{ e.value = '{to_date}'; e.dispatchEvent(new Event('input')); }}"
                        driver.execute_script(script)

                time.sleep(0.8)

                # 2) Set Reportable = YES
                reportable_set = False
                try:
                    possible_dropdown_ids = ['ddlReportable', 'rptable', 'ddlReport', 'ddlreportable']
                    for d in possible_dropdown_ids:
                        try:
                            sel = Select(driver.find_element(By.ID, d))
                            for option in sel.options:
                                if option.text.strip().upper() == "YES":
                                    sel.select_by_visible_text(option.text)
                                    reportable_set = True
                                    break
                            if reportable_set:
                                break
                        except:
                            continue
                except:
                    pass

                if not reportable_set:
                    radio_candidates = [
                        "//input[@type='radio' and (translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='rpjudgey' or translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='rpjudge') and (translate(@value,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ')='Y' or translate(@value,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ')='YES')]",
                        "//input[contains(translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'report') and (@value='Y' or @value='Yes' or @value='YES')]",
                        "//input[contains(@name,'report') and (@value='Y' or @value='Yes' or @value='YES')]"
                    ]
                    for xpath in radio_candidates:
                        try:
                            radios = driver.find_elements(By.XPATH, xpath)
                            if radios:
                                try:
                                    radios[0].click()
                                    reportable_set = True
                                    break
                                except:
                                    driver.execute_script("arguments[0].click();", radios[0])
                                    reportable_set = True
                                    break
                        except:
                            continue

                time.sleep(0.5)

                # 3) Find Submit button
                search_btn = None
                search_candidates = [
                    ("id", "btncasedetail1_1"),
                    ("id", "btnSearch"),
                    ("id", "btncasedetail1"),
                    ("xpath", "//button[contains(translate(text(),'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'SEARCH')]"),
                    ("xpath", "//input[@type='button' and (contains(translate(@value,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'SEARCH') or contains(translate(@value,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'SUBMIT'))]"),
                    ("xpath", "//a[contains(translate(text(),'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'SEARCH')]")
                ]
                for typ, val in search_candidates:
                    try:
                        if typ == "id":
                            el = driver.find_element(By.ID, val)
                        else:
                            el = driver.find_element(By.XPATH, val)
                        if el:
                            search_btn = el
                            break
                    except:
                        continue

                # 4) Enhanced captcha handling
                max_attempts = 3
                for attempt in range(max_attempts):
                    print(f"\nCaptcha attempt {attempt+1}/{max_attempts}")
                    
                    # Try PaddleOCR first
                    captcha_text = ""
                    if PADDLEOCR_AVAILABLE and self.ocr is not None:
                        captcha_text = self.extract_captcha_from_base64(driver)
                    
                    # Fallback to manual input
                    if not captcha_text:
                        captcha_text = self.solve_captcha_manual(driver)

                    if not captcha_text:
                        print("No captcha text provided; aborting search attempt.")
                        return False

                    # Find captcha input field
                    captcha_input = None
                    captcha_candidate_ids = ['txtCaptcha', 'captcha', 'Captcha', 'captchacode', 'captcha_code', 'security_code']
                    for cid in captcha_candidate_ids:
                        try:
                            captcha_input = driver.find_element(By.ID, cid)
                            break
                        except:
                            try:
                                captcha_input = driver.find_element(By.NAME, cid)
                                break
                            except:
                                continue

                    if not captcha_input:
                        try:
                            elems = driver.find_elements(By.XPATH, "//input[contains(@placeholder,'captcha') or contains(translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'captcha')]")
                            if elems:
                                captcha_input = elems[0]
                        except:
                            pass

                    if not captcha_input:
                        print("Could not find captcha input field on page.")
                        try:
                            driver.save_screenshot("no_captcha_input.png")
                        except:
                            pass
                        return False

                    # Enter captcha text
                    try:
                        captcha_input.clear()
                        captcha_input.send_keys(captcha_text)
                    except Exception as e:
                        try:
                            driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input'));", captcha_input, captcha_text)
                        except:
                            print(f"Failed to enter captcha: {e}")
                            continue

                    time.sleep(0.3)

                    # Click search button
                    if not search_btn:
                        print("Search button not found; cannot submit.")
                        return False

                    try:
                        driver.execute_script("arguments[0].click();", search_btn)
                    except:
                        try:
                            search_btn.click()
                        except Exception as e:
                            print(f"Failed to click search button: {e}")
                            return False

                    time.sleep(2)

                    # Wait for results
                    try:
                        WebDriverWait(driver, 12).until(
                            EC.any_of(
                                EC.presence_of_element_located((By.ID, "grdJudgement")),
                                EC.presence_of_element_located((By.XPATH, "//table[contains(@id,'grd') or contains(@class,'grid') or contains(@class,'Grd')]")),
                                EC.presence_of_element_located((By.XPATH, "//*[contains(translate(text(),'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'NO RECORD')]"))
                            )
                        )
                    except:
                        page_text = driver.page_source.lower()
                        if "no record" in page_text or "no records found" in page_text or "no record found" in page_text:
                            print("No records found for the given date range.")
                            return True
                        time.sleep(2)

                    # Check for invalid captcha
                    page_src = driver.page_source.lower()
                    if any(err in page_src for err in ["invalid captcha", "incorrect captcha", "wrong captcha", "captcha mismatch"]):
                        print("Captcha appears invalid, retrying...")
                        # Try to refresh captcha
                        try:
                            refresh_buttons = driver.find_elements(By.XPATH, "//a[contains(@onclick,'getCaptcha')]")
                            if refresh_buttons:
                                try:
                                    driver.execute_script("arguments[0].click();", refresh_buttons[0])
                                    time.sleep(1)  # Wait for new captcha to load
                                except:
                                    pass
                        except:
                            pass
                        continue

                    print("Form submitted and page changed (results or no-record detected).")
                    return True

                print("All captcha attempts exhausted and failed.")
                return False

            except Exception as e:
                self.logger.error(f"Error filling form: {e}")
                try:
                    driver.save_screenshot("form_failure.png")
                except:
                    pass
                return False
        
    def extract_judgment(self, driver: webdriver.Chrome) -> List[Dict]:
            """Extract judgment data from the results page"""
            judgments: List[Dict] = []
            try:
                self.logger.info("Extracting judgment data...")

                wait = WebDriverWait(driver, 15)
                try:
                    wait.until(lambda d: (
                        d.find_elements(By.TAG_NAME, "table")
                        or d.find_elements(By.XPATH, "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]")
                        or d.find_elements(By.XPATH, "//tr[count(td) > 1]")
                        or "no record" in d.page_source.lower()
                    ))
                except Exception:
                    self.logger.warning("Timed out waiting for obvious result elements; continuing to search DOM.")

                page_source = driver.page_source.lower()
                if "no record" in page_source or "no records found" in page_source or "no record found" in page_source:
                    self.logger.info("Detected 'no records' text on page.")
                    return []

                # Find the best table for results
                tables = driver.find_elements(By.TAG_NAME, "table")
                self.logger.info(f"Found {len(tables)} <table> elements - scoring to find results table...")
                results_table = None
                best_score = 0

                keywords = ["case", "judgment", "date", "party", "petitioner", "respondent", "bench", "case no", "s.no", "serial"]

                for table in tables:
                    try:
                        text = table.text.lower()
                        score = sum(1 for kw in keywords if kw in text)
                        try:
                            if table.find_elements(By.XPATH, ".//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]"):
                                score += 2
                        except Exception:
                            pass
                        if score > best_score:
                            best_score = score
                            results_table = table
                    except Exception:
                        continue

                if results_table is not None and best_score > 0:
                    self.logger.info(f"Selected table with score {best_score} as results table.")
                    # Extract headers
                    headers = []
                    try:
                        header_row = results_table.find_element(By.TAG_NAME, "tr")
                        ths = header_row.find_elements(By.TAG_NAME, "th")
                        if not ths:
                            ths = header_row.find_elements(By.TAG_NAME, "td")
                        for th in ths:
                            h = th.text.strip()
                            headers.append(h if h else f"Column_{len(headers)+1}")
                    except Exception:
                        try:
                            first_data_row = results_table.find_element(By.XPATH, ".//tr[count(td)>0][2]")
                            cell_count = len(first_data_row.find_elements(By.TAG_NAME, "td"))
                            headers = [f"Column_{i+1}" for i in range(cell_count)]
                        except Exception:
                            headers = []

                    # Parse data rows
                    rows = results_table.find_elements(By.XPATH, ".//tr[position()>1]")
                    for r in rows:
                        try:
                            cells = r.find_elements(By.TAG_NAME, "td")
                            if not cells:
                                continue
                            row_data: Dict[str, str] = {}
                            for i, c in enumerate(cells):
                                colname = headers[i] if i < len(headers) else f"Column_{i+1}"
                                row_data[colname] = c.text.strip()
                            
                            # Find PDF link
                            pdf_el = None
                            try:
                                pdf_el = r.find_element(By.XPATH, ".//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]")
                            except Exception:
                                try:
                                    pdf_el = r.find_element(By.XPATH, ".//a[contains(translate(text(),'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'VIEW') or contains(translate(text(),'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'DOWNLOAD')]")
                                except Exception:
                                    pdf_el = None
                            
                            row_data["pdf_url"] = pdf_el.get_attribute("href") or "" if pdf_el else ""
                            row_data["judgment_id"] = self.generate_judgment_id(row_data)
                            judgments.append(row_data)
                        except Exception as e:
                            self.logger.debug(f"Skipping a row due to parse error: {e}")
                    
                    self.logger.info(f"Extracted {len(judgments)} rows from selected table.")
                    return judgments

                # Fallback methods for other table structures
                self.logger.info("No clear results table found - trying fallback methods...")
                
                # Try loose rows
                loose_rows = driver.find_elements(By.XPATH, "//tr[count(td) > 1]")
                if loose_rows:
                    for r in loose_rows:
                        try:
                            cells = r.find_elements(By.TAG_NAME, "td")
                            if not cells or len(cells) < 2:
                                continue
                            row_data = {}
                            for i, c in enumerate(cells):
                                row_data[f"Column_{i+1}"] = c.text.strip()
                            
                            try:
                                pdf_el = r.find_element(By.XPATH, ".//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]")
                                row_data["pdf_url"] = pdf_el.get_attribute("href") or ""
                            except Exception:
                                row_data['pdf_url'] = ""
                            
                            row_data["judgment_id"] = self.generate_judgment_id(row_data)
                            judgments.append(row_data)
                        except Exception:
                            continue
                    
                    if judgments:
                        self.logger.info(f"Extracted {len(judgments)} rows from loose parsing.")
                        return judgments

                # Final fallback - standalone PDF links
                self.logger.info("Looking for standalone PDF links...")
                pdf_links = driver.find_elements(By.XPATH, "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]")
                if pdf_links:
                    for a in pdf_links:
                        try:
                            title = a.text.strip() or a.get_attribute('title') or a.get_attribute('aria-label') or ""
                            href = a.get_attribute('href') or ""
                            row_data = {"title": str(title or ""), "pdf_url": str(href or "")}
                            row_data["judgment_id"] = self.generate_judgment_id(row_data)
                            judgments.append(row_data)
                        except Exception:
                            continue
                    
                    self.logger.info(f"Found {len(judgments)} standalone PDF links.")
                    return judgments

                self.logger.warning("No judgement data found on the page.")
                try:
                    driver.save_screenshot("no_results.png")
                    self.logger.info("Saved debug screenshot as 'no_results.png'")
                except Exception:
                    pass
                return []

            except Exception as e:
                self.logger.error(f"Error extracting judgments: {e}", exc_info=True)
                try:
                    driver.save_screenshot("extraction_error.png")
                except:
                    pass
                return []

    def download_pdf(self, url, filename):
            """Download PDF file"""
            try:
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                filepath = self.pdf_dir / filename
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if filepath.exists() and filepath.stat().st_size > 1024:
                    return True
                
            except Exception as e:
                self.logger.error(f"Download failed for {filename}: {e}")
            
            return False
        
    def generate_pdf_filename(self, judgment_data):
            """Generate safe PDF filename"""
            identifier = "judgment"
            
            for key, value in judgment_data.items():
                if value and any(word in key.lower() for word in ['case', 'number', 'serial']):
                    identifier = str(value).replace('/', '_').replace('\\', '_')[:30]
                    break
            
            safe_name = ''.join(c for c in identifier if c.isalnum() or c in '._-')
            return f"{safe_name}_{int(time.time())}.pdf"
        
    def process_judgments(self, judgments):
            """Process judgments (download PDFs, filter duplicates)"""
            processed = []
            new_count = 0
            
            for judgment in judgments:
                judgment_id = judgment['judgment_id']
                
                if judgment_id in self.state["downloaded_ids"]:
                    continue
                
                # Download PDF if available
                if judgment.get('pdf_url'):
                    filename = self.generate_pdf_filename(judgment)
                    
                    if self.download_pdf(judgment['pdf_url'], filename):
                        judgment['pdf_filename'] = filename
                        judgment['download_status'] = "Success"
                        print(f"✅ Downloaded: {filename}")
                    else:
                        judgment['pdf_filename'] = "Failed"
                        judgment['download_status'] = "Failed"
                        print(f"❌ Failed: {filename}")
                else:
                    judgment['pdf_filename'] = "No_URL"
                    judgment['download_status'] = "No_URL"
                
                # Mark as processed
                self.state["downloaded_ids"].add(judgment_id)
                processed.append(judgment)
                new_count += 1
            
            print(f"✅ Processed {new_count} new judgments")
            return processed
        
    def save_to_csv(self, judgments):
            """Save judgments to CSV"""
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
                # Align columns
                all_columns = set(existing_df.columns) | set(new_df.columns)
                for col in all_columns:
                    if col not in existing_df.columns:
                        existing_df[col] = ""
                    if col not in new_df.columns:
                        new_df[col] = ""
                
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Remove duplicates
            if 'judgment_id' in combined_df.columns:
                before = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['judgment_id'], keep='last')
                after = len(combined_df)
                if before != after:
                    print(f"Removed {before - after} duplicates")
            
            # Save to CSV
            combined_df.to_csv(self.csv_file, index=False)
            print(f"✅ Saved {len(combined_df)} total records to CSV")
        
    def run_scraper(self, days_back=10):
            """Main scraper execution"""
            print(f"\n{'='*60}")
            print("RAJASTHAN HIGH COURT JUDGMENT SCRAPER")
            print(f"{'='*60}")
            
            # Calculate date range
            india_time = get_india_time()
            end_date = india_time.date()
            start_date = end_date - timedelta(days=days_back)
            
            from_date_str = start_date.strftime("%d/%m/%Y")
            to_date_str = end_date.strftime("%d/%m/%Y")
            
            print(f"Date range: {from_date_str} to {to_date_str}")
            print(f"Download directory: {self.download_dir}")
            print(f"OCR Engine: {'PaddleOCR' if PADDLEOCR_AVAILABLE else 'Manual only'}")
            
            # Setup driver
            print("Setting up Chrome driver...")
            driver = self.setup_driver()
            
            try:
                # Load website
                print("Loading website...")
                driver.get(self.base_url)
                
                # Take screenshot for debugging
                driver.save_screenshot("page_loaded.png")
                print("Screenshot saved as 'page_loaded.png'")
                
                # Wait for page to stabilize
                time.sleep(5)
                
                print(f"Page title: {driver.title}")
                
                # Debug form elements
                print("Debugging form elements...")
                inputs = driver.find_elements(By.TAG_NAME, "input")
                print(f"Found {len(inputs)} input elements:")
                for i, inp in enumerate(inputs[:10]):
                    try:
                        print(f"  {i+1}: type='{inp.get_attribute('type')}', "
                            f"id='{inp.get_attribute('id')}', "
                            f"name='{inp.get_attribute('name')}', "
                            f"placeholder='{inp.get_attribute('placeholder')}'")
                    except:
                        pass
                
                buttons = driver.find_elements(By.TAG_NAME, "button")
                print(f"\nFound {len(buttons)} button elements:")
                for i, btn in enumerate(buttons):
                    try:
                        print(f"  {i+1}: text='{btn.text}', id='{btn.get_attribute('id')}'")
                    except:
                        pass
                
                # Fill form and search
                if not self.fill_form_and_search(driver, from_date_str, to_date_str):
                    print("Failed to submit search form")
                    driver.save_screenshot("form_failure.png")
                    print("Failure screenshot saved as 'form_failure.png'")
                    return []
                
                # Extract judgments
                print("Extracting judgment data...")
                judgments = self.extract_judgment(driver)
                
                if not judgments:
                    print("No judgments found")
                    driver.save_screenshot("no_results.png")
                    print("No results screenshot saved as 'no_results.png'")
                    return []
                
                # Process judgments
                print("Processing judgments...")
                processed_judgments = self.process_judgments(judgments)
                
                # Save to CSV
                if processed_judgments:
                    print("Saving to CSV...")
                    self.save_to_csv(processed_judgments)
                
                # Save state
                self.save_state()
                
                return processed_judgments
                
            except Exception as e:
                self.logger.error(f"Scraping error: {e}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                try:
                    driver.save_screenshot("error_screenshot.png")
                    print("Error screenshot saved as 'error_screenshot.png'")
                except:
                    pass
                return []
            
            finally:
                driver.quit()

    def show_results(self):
            """Display results summary"""
            if self.csv_file.exists():
                df = pd.read_csv(self.csv_file)
                
                print(f"\n{'='*50}")
                print("RESULTS SUMMARY")
                print(f"{'='*50}")
                print(f"📊 Total judgments in database: {len(df)}")
                
                if 'download_status' in df.columns:
                    print("\n📈 Download Statistics:")
                    status_counts = df['download_status'].value_counts()
                    for status, count in status_counts.items():
                        print(f"   {status}: {count}")
                
                print(f"\n📄 CSV file: {self.csv_file}")
                print(f"📁 PDF directory: {self.pdf_dir}")
                
                if len(df) > 0:
                    print(f"\n🔍 Sample data:")
                    sample = df.head(1)
                    for col in sample.columns:
                        value = sample[col].iloc[0]
                        if value:
                            print(f"   {col}: {str(value)[:60]}...")
            else:
                print("No results file found")


    def main(self) -> None:
        """Main execution function"""
        try:
            print("Rajasthan High Court Judgment Scraper with PaddleOCR")
            print("=" * 60)
            
            if not PADDLEOCR_AVAILABLE:
                print("WARNING: PaddleOCR is not available!")
                print("To install PaddleOCR, run:")
                print("  pip install paddleocr")
                print("  pip install opencv-python")
                print("Note: Without PaddleOCR, you'll need to manually enter captcha codes.")
                print()
            
            # Create scraper instance
            scraper = RajasthanHCScraper(headless=False)  # Set to True for headless mode
            # scraper = RajasthanHCScraper(headless=False)  # For debugging, set headless=False
            numerical_captcha_extractor(scraper)
            # Run scraper
            judgments = scraper.run_scraper(days_back=10)
            
            # Show results
            scraper.logger.info(f"Scraped {len(judgments)} judgments")
            scraper.show_results()
            
            print(f"\n✅ Scraping completed!")
            print(f"📋 Found {len(judgments)} new judgments this run")
            
            if judgments:
                print("\n🔍 Sample judgment data:")
                for key, value in list(judgments[0].items())[:5]:
                    print(f"   {key}: {value}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Scraping interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    RajasthanHCScraper().main()