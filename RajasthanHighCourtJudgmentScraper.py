#!/usr/bin/env python3
"""
Fixed Rajasthan High Court Judgment Scraper
Handles the current website structure and fixes captcha detection issues
Optimized for performance and reliability
"""

import os
import csv
import json
import time
import hashlib
import requests
import logging
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
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
from selenium.common.exceptions import (TimeoutException, NoSuchElementException, 
                                       StaleElementReferenceException, WebDriverException)
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rajasthan_hc_scraper.log"),
        logging.StreamHandler()
    ]
)

# Try to import pytesseract with error handling
try:
    import pytesseract
    # For Windows users, set the correct path if it exists
    if os.name == 'nt' and os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    logging.warning("Tesseract not available. Captcha solving will be manual.")
    TESSERACT_AVAILABLE = False

# Try to import the advanced captcha solver
try:
    from Advanced_OCR_Captcha_Solver import AdvancedCaptchaOCR
    ADVANCED_OCR_AVAILABLE = True
except ImportError:
    ADVANCED_OCR_AVAILABLE = False

class FixedRajasthanHCScraper:
    def __init__(self, download_dir: str = "rajasthan_hc_judgments", headless: bool = False, debug: bool = False):
        self.base_url = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/"
        self.download_dir = Path(download_dir)
        self.pdf_dir = self.download_dir / "pdfs"
        self.csv_file = self.download_dir / "judgments.csv"
        self.state_file = self.download_dir / "scraper_state.json"
        self.debug = debug
        self.logger = logging.getLogger('RajasthanHCScraper')
        
        # Create directories
        self.download_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.downloaded_judgments = self.load_state()
        
        # Setup Chrome options
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Performance optimizations
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-popup-blocking")
        
        # Set download preferences
        prefs = {
            "download.default_directory": str(self.pdf_dir.absolute()),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_settings.popups": 0,
            "safebrowsing.enabled": False  # Disable safe browsing checks for faster downloads
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
        
        # Initialize captcha solver
        if ADVANCED_OCR_AVAILABLE:
            self.captcha_solver = AdvancedCaptchaOCR(debug_mode=debug)
            self.logger.info("Using Advanced OCR Captcha Solver")
        else:
            self.captcha_solver = None
            self.logger.warning("Advanced OCR Captcha Solver not available")
            
        # Performance metrics
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_judgments": 0,
            "new_judgments": 0,
            "captcha_attempts": 0,
            "captcha_success": 0,
            "download_success": 0,
            "download_failed": 0,
            "errors": 0
        }
        
    def load_state(self) -> Dict:
        """Load previously downloaded judgment IDs and metadata"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    state["downloaded_ids"] = set(state.get("downloaded_ids", []))
                    self.logger.info(f"Loaded state with {len(state['downloaded_ids'])} previously downloaded judgments")
                    return state
            except Exception as e:
                self.logger.error(f"Error loading state: {e}", exc_info=True)
        self.logger.info("No previous state found, starting fresh")
        return {"downloaded_ids": set(), "last_run_date": None}
    
    def save_state(self):
        """Save current state to file"""
        state_to_save = {
            "downloaded_ids": list(self.downloaded_judgments["downloaded_ids"]),
            "last_run_date": self.downloaded_judgments["last_run_date"]
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2)
            self.logger.info(f"Saved state with {len(self.downloaded_judgments['downloaded_ids'])} downloaded judgments")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}", exc_info=True)
    
    def generate_judgment_id(self, judgment_data: Dict) -> str:
        """Generate unique ID for judgment based on key fields"""
        # Use multiple fields to create a more unique ID
        id_parts = []
        key_fields = ['case_number', 'petitioner', 'respondent', 'judgment_date', 'judge_name']
        
        # Prioritize specific key fields if available
        for key in key_fields:
            if key in judgment_data and judgment_data[key] and str(judgment_data[key]).strip():
                id_parts.append(str(judgment_data[key]).strip())
        
        # If we don't have enough key fields, use the first 5 available fields
        if len(id_parts) < 3:
            for key, value in list(judgment_data.items())[:5]:  # First 5 fields
                if value and str(value).strip() and str(value).strip() not in id_parts:
                    id_parts.append(str(value).strip())
        
        id_string = "|".join(id_parts) if id_parts else str(hash(str(judgment_data)))
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def enhance_captcha_image(self, image_path: str) -> str:
        """Enhanced captcha image processing using advanced techniques"""
        try:
            # Load image with PIL first for better compatibility
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return ""
                
            # Load with PIL
            pil_img = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_img.mode not in ['RGB', 'L']:
                pil_img = pil_img.convert('RGB')
            
            # Resize image for better OCR (minimum 150px width)
            width, height = pil_img.size
            if width < 150 or height < 50:
                scale_factor = max(150/width, 50/height, 3)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
            
            # Convert to OpenCV format
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Create a list to store processed images with their names
            processed_images = []
            
            # Method 1: Enhanced contrast + Otsu threshold
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Enhanced_Otsu", otsu_thresh))
            
            # Method 2: Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, gaussian_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Gaussian_Blur", gaussian_thresh))
            
            # Method 3: Median filter (good for removing noise)
            median_filtered = cv2.medianBlur(gray, 3)
            _, median_thresh = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Median_Filter", median_thresh))
            
            # Method 4: Bilateral filter (preserves edges)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Bilateral_Filter", bilateral_thresh))
            
            # Method 5: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_images.append(("Adaptive_Threshold", adaptive))
            
            # Method 6: Morphological operations
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            processed_images.append(("Morphological", morph))
            
            # Method 7: Edge detection + dilation
            edges = cv2.Canny(gray, 50, 150)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            inverted_edges = cv2.bitwise_not(dilated)
            processed_images.append(("Edge_Enhanced", inverted_edges))
            
            # Method 8: Inverted images (sometimes captcha has dark background)
            inverted_otsu = cv2.bitwise_not(otsu_thresh)
            processed_images.append(("Inverted_Otsu", inverted_otsu))
            
            # Method 9: High contrast processing
            high_contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=-50)
            _, high_contrast_thresh = cv2.threshold(high_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("High_Contrast", high_contrast_thresh))
            
            # Method 10: Erosion + Dilation (remove small noise)
            eroded = cv2.erode(otsu_thresh, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            processed_images.append(("Erode_Dilate", dilated))
            
            # Save processed images for debugging if needed
            debug_dir = "captcha_debug"
            if self.debug and not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
                for name, img in processed_images:
                    cv2.imwrite(f"{debug_dir}/{name}.png", img)
            
            # Try OCR with multiple configurations on all processed images
            results = []
            
            # Different OCR configurations
            configs = [
                r'--oem 3 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 6 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 13 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 1 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 1 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 0 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 0 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 0 --psm 6 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 0 --psm 13 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 10 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 11 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            for name, processed_img in processed_images:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(processed_img, config=config).strip()
                        # Clean the text
                        cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                        
                        # Calculate confidence score based on length and character composition
                        confidence = 0
                        if 4 <= len(cleaned_text) <= 8:  # Most captchas are 4-8 characters
                            confidence += 0.5
                            
                            # Check if it has a good mix of letters and numbers
                            has_letters = any(c.isalpha() for c in cleaned_text)
                            has_numbers = any(c.isdigit() for c in cleaned_text)
                            
                            if has_letters and has_numbers:  # Many captchas mix letters and numbers
                                confidence += 0.3
                            
                            # Avoid results with too many repeated characters
                            if len(set(cleaned_text)) >= len(cleaned_text) * 0.75:  # At least 75% unique chars
                                confidence += 0.2
                            
                            results.append((cleaned_text, confidence, name, config))
                    except Exception as e:
                        continue
            
            # Sort results by confidence
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return the highest confidence result if available
            if results:
                best_result = results[0]
                print(f"OCR success: {best_result[0]} (confidence: {best_result[1]:.2f}, method: {best_result[2]}, config: {best_result[3][:20]}...)")
                return best_result[0]
            
            return ""
            
        except Exception as e:
            print(f"Error in captcha enhancement: {e}")
            return ""
    
    def solve_captcha_ocr(self, driver, attempt=0) -> str:
        """Advanced OCR captcha solving with improved image detection and processing
        
        Args:
            driver: Selenium WebDriver instance
            attempt: Current attempt number (used to try different strategies)
            
        Returns:
            str: The solved captcha text or empty string if failed
        """
        self.metrics["captcha_attempts"] += 1
        
        # Use the advanced captcha solver if available
        if ADVANCED_OCR_AVAILABLE and self.captcha_solver:
            self.logger.info("Using Advanced OCR Captcha Solver")
            return self._solve_with_advanced_ocr(driver, attempt)
            
        if not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseract OCR not available")
            return ""
        
        try:
            print("Looking for captcha image...")
            
            # Create directory for captcha images
            captcha_dir = os.path.join(os.getcwd(), "captcha_images")
            os.makedirs(captcha_dir, exist_ok=True)
            
            # Generate unique filename for this attempt
            timestamp = int(time.time())
            captcha_filename = f"captcha_{timestamp}_{attempt}.png"
            captcha_path = os.path.join(captcha_dir, captcha_filename)
            
            # Multiple ways to find captcha image with expanded selectors
            captcha_selectors = [
                "//img[contains(@src, 'captcha')]",
                "//img[contains(@src, 'Captcha')]",
                "//img[contains(@src, 'CAPTCHA')]",
                "//img[contains(@id, 'captcha')]",
                "//img[contains(@id, 'Captcha')]",
                "//img[contains(@class, 'captcha')]",
                "//img[contains(@alt, 'captcha')]",
                "//img[contains(@title, 'captcha')]",
                "//img[contains(@name, 'captcha')]",
                "//img[contains(@name, 'Captcha')]",
                "//img[contains(@src, 'jcaptcha')]",
                "//img[contains(@src, 'securimage')]",
                "//img[contains(@src, 'random')]",
                "//img[contains(@src, 'verify')]",
                "//img[contains(@src, 'verification')]"
            ]
            
            # Rotate through different selector strategies based on attempt number
            if attempt > 0:
                # Reorder selectors to try different ones first on subsequent attempts
                captcha_selectors = captcha_selectors[attempt % len(captcha_selectors):] + captcha_selectors[:attempt % len(captcha_selectors)]
            
            captcha_img = None
            used_selector = None
            
            # Try standard selectors first
            for selector in captcha_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        for i, img in enumerate(elements):
                            # On subsequent attempts, try different images if multiple match
                            if i == (attempt % max(1, len(elements))):
                                captcha_img = img
                                used_selector = selector
                                print(f"Found captcha with selector: {selector} (image {i+1} of {len(elements)})")
                                break
                        if captcha_img:
                            break
                except Exception as e:
                    print(f"Error with selector {selector}: {e}")
                    continue
            
            # If standard selectors failed, try proximity-based detection
            if not captcha_img:
                print("No captcha image found with standard selectors, trying proximity detection...")
                try:
                    # Try to find captcha input field first
                    captcha_input = None
                    input_selectors = [
                        "//input[contains(@id, 'captcha') or contains(@id, 'Captcha')]",
                        "//input[contains(@name, 'captcha') or contains(@name, 'Captcha')]",
                        "//input[contains(@placeholder, 'captcha') or contains(@placeholder, 'enter text')]",
                        "//input[contains(@class, 'captcha')]"
                    ]
                    
                    for selector in input_selectors:
                        inputs = driver.find_elements(By.XPATH, selector)
                        if inputs:
                            captcha_input = inputs[0]
                            print(f"Found captcha input with selector: {selector}")
                            break
                    
                    if captcha_input:
                        # Look for images near the captcha input
                        # Get the location of the input field
                        input_location = captcha_input.location
                        
                        # Find all images on the page
                        all_images = driver.find_elements(By.TAG_NAME, "img")
                        
                        # Sort images by proximity to the input field
                        def distance(img):
                            try:
                                img_loc = img.location
                                return ((img_loc['x'] - input_location['x'])**2 + 
                                        (img_loc['y'] - input_location['y'])**2)**0.5
                            except:
                                return 99999
                        
                        nearby_images = sorted(all_images, key=distance)
                        
                        # Use the closest image or the one corresponding to the attempt number
                        if nearby_images:
                            img_index = min(attempt, len(nearby_images)-1)
                            captcha_img = nearby_images[img_index]
                            print(f"Using proximity-based image detection (image {img_index+1} of {len(nearby_images)})")
                    else:
                        # If we can't find the input field, try using small images as they're often captchas
                        small_images = []
                        for img in driver.find_elements(By.TAG_NAME, "img"):
                            try:
                                size = img.size
                                if 20 <= size['width'] <= 200 and 10 <= size['height'] <= 80:
                                    small_images.append(img)
                            except:
                                continue
                        
                        if small_images:
                            img_index = attempt % len(small_images)
                            captcha_img = small_images[img_index]
                            print(f"Using size-based image detection (image {img_index+1} of {len(small_images)})")
                except Exception as e:
                    print(f"Error in proximity detection: {e}")
            
            # Last resort: try to take a screenshot of a specific region where captchas often appear
            if not captcha_img:
                print("Could not locate captcha image, trying full page screenshot...")
                try:
                    # Take full page screenshot
                    driver.save_screenshot(captcha_path)
                    print(f"Saved full page screenshot to {captcha_path}")
                    
                    # Try to process the full image - sometimes OCR can still find the captcha
                    captcha_text = self.enhance_captcha_image(captcha_path)
                    if captcha_text:
                        print(f"Found captcha text from full page: {captcha_text}")
                        return captcha_text
                    else:
                        return ""
                except Exception as e:
                    print(f"Error taking full screenshot: {e}")
                    return ""
            
            # Take screenshot of captcha element
            try:
                captcha_img.screenshot(captcha_path)
                print(f"Captcha image saved to {captcha_path}")
                
                # Get image attributes for debugging
                try:
                    img_src = captcha_img.get_attribute("src") or "unknown"
                    img_size = f"{captcha_img.size['width']}x{captcha_img.size['height']}"
                    print(f"Captcha image details - src: {img_src[:50]}{'...' if len(img_src) > 50 else ''}, size: {img_size}")
                except:
                    pass
                
                # Enhanced processing
                captcha_text = self.enhance_captcha_image(captcha_path)
                
                # If OCR failed, try direct image URL download if it's a remote image
                if not captcha_text:
                    try:
                        img_src = captcha_img.get_attribute("src")
                        if img_src and (img_src.startswith("http") or img_src.startswith("data:image")):
                            print("Trying direct image download...")
                            direct_path = os.path.join(captcha_dir, f"direct_{timestamp}.png")
                            
                            if img_src.startswith("data:image"):
                                # Handle base64 encoded image
                                import base64
                                img_data = img_src.split(',')[1]
                                with open(direct_path, "wb") as f:
                                    f.write(base64.b64decode(img_data))
                            else:
                                # Handle remote URL
                                import urllib.request
                                urllib.request.urlretrieve(img_src, direct_path)
                            
                            print(f"Downloaded image directly to {direct_path}")
                            captcha_text = self.enhance_captcha_image(direct_path)
                    except Exception as e:
                        print(f"Error in direct image download: {e}")
                
                return captcha_text
            except Exception as e:
                print(f"Error taking captcha screenshot: {e}")
                return ""
            
        except Exception as e:
            print(f"Error in OCR captcha solving: {e}")
            import traceback
            traceback.print_exc()
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
    
    def solve_captcha(self, driver, max_attempts=5) -> str:
        """Comprehensive captcha solving with multiple attempts and validation
        
        Args:
            driver: Selenium WebDriver instance
            max_attempts: Maximum number of attempts to solve captcha automatically
            
        Returns:
            str: The solved captcha text
        """
        print("Solving captcha...")
        
        # Create a directory for captcha images if it doesn't exist
        captcha_dir = os.path.join(os.getcwd(), "captcha_images")
        os.makedirs(captcha_dir, exist_ok=True)
        
        # Track captcha results for consistency checking and reuse
        captcha_results = []
        
        # First try OCR with multiple attempts
        if TESSERACT_AVAILABLE:
            for attempt in range(max_attempts):
                print(f"\nCaptcha OCR attempt {attempt+1}/{max_attempts}")
                
                # Try refreshing the captcha image on subsequent attempts
                if attempt > 0:
                    try:
                        # Look for captcha refresh button with multiple selectors
                        refresh_button = None
                        for selector in ["btnRefresh", "refresh", "captchaRefresh"]:
                            try:
                                refresh_button = driver.find_element(By.ID, selector)
                                break
                            except NoSuchElementException:
                                continue
                        
                        if refresh_button:
                            refresh_button.click()
                            time.sleep(1)
                    except Exception as e:
                        print(f"Error refreshing captcha: {e}")
                        # Look for captcha refresh button with multiple selectors
                        refresh_selectors = [
                            "//img[contains(@src, 'refresh')]",
                            "//img[contains(@src, 'reload')]",
                            "//img[contains(@alt, 'refresh')]",
                            "//img[contains(@title, 'refresh')]",
                            "//button[contains(@id, 'refresh')]",
                            "//a[contains(@id, 'refresh')]",
                            "//i[contains(@class, 'refresh')]",
                            "//i[contains(@class, 'reload')]",
                            "//i[contains(@class, 'fa-sync')]",
                            "//i[contains(@class, 'fa-redo')]"
                        ]
                        
                        refresh_clicked = False
                        for selector in refresh_selectors:
                            try:
                                elements = driver.find_elements(By.XPATH, selector)
                                if elements:
                                    elements[0].click()
                                    print(f"Clicked captcha refresh button with selector: {selector}")
                                    refresh_clicked = True
                                    time.sleep(1)  # Wait for new captcha to load
                                    break
                            except:
                                continue
                        if not refresh_clicked:
                            # Try alternative selectors
                            refresh_selectors = [
                                "//img[contains(@src, 'refresh')]",
                                "//img[contains(@title, 'refresh')]",
                                "//img[contains(@alt, 'refresh')]",
                                "//a[contains(@onclick, 'captcha')]",
                                "//a[contains(@href, 'javascript') and contains(@href, 'captcha')]"
                            ]
                            
                            for selector in refresh_selectors:
                                try:
                                    refresh_buttons = driver.find_elements(By.XPATH, selector)
                                    if refresh_buttons:
                                        refresh_buttons[0].click()
                                        print(f"Refreshed captcha on attempt {attempt+1}")
                                        time.sleep(1)  # Wait for new captcha to load
                                        break
                                except Exception as e:
                                    print(f"Could not refresh captcha with selector {selector}: {e}")
                    
                    # Try OCR with the current attempt number
                    try:
                        ocr_result = self.solve_captcha_ocr(driver, attempt)
                        
                        # Clean up the result - remove non-alphanumeric characters
                        if ocr_result:
                            ocr_result = ''.join(c for c in ocr_result if c.isalnum())
                        
                        # Validate captcha text format
                        if ocr_result and 4 <= len(ocr_result) <= 8:
                            # Check if it has a good mix of characters (most captchas do)
                            has_letters = any(c.isalpha() for c in ocr_result)
                            has_numbers = any(c.isdigit() for c in ocr_result)
                            
                            # Add to results list for consistency checking
                            captcha_results.append(ocr_result)
                            
                            # Most captchas have both letters and numbers
                            if has_letters and has_numbers:
                                print(f"Solved captcha automatically (attempt {attempt+1}): {ocr_result}")
                                return ocr_result
                            else:
                                print(f"Captcha text format suspicious (attempt {attempt+1}): {ocr_result}")
                                
                                # If we've seen this result multiple times, it might be correct despite format
                                if captcha_results.count(ocr_result) >= 2:
                                    print(f"Using consistent captcha result: {ocr_result}")
                                    return ocr_result
                    except Exception as e:
                        print(f"OCR attempt {attempt+1} failed: {e}")
                        time.sleep(1)  # Brief pause before next attempt
            
            # Check if we have any consistent results before falling back to manual
            if captcha_results:
                from collections import Counter
                result_counts = Counter(captcha_results)
                most_common = result_counts.most_common(1)[0]
                if most_common[1] >= 2 and most_common[0]:
                    print(f"Using most consistent captcha result: {most_common[0]} (appeared {most_common[1]} times)")
                    return most_common[0]
                else:
                    # If no consistent result but we have at least one result, use the last one
                    print(f"Using last OCR result: {captcha_results[-1]}")
                    return captcha_results[-1]
        else:
            print("Tesseract OCR not available. Falling back to manual input.")
        
        # Take a screenshot of the current page for debugging
        try:
            debug_path = os.path.join(os.getcwd(), "debug_screenshots")
            os.makedirs(debug_path, exist_ok=True)
            timestamp = int(time.time())
            screenshot_path = os.path.join(debug_path, f"captcha_page_{timestamp}.png")
            driver.save_screenshot(screenshot_path)
            print(f"Saved debug screenshot to {screenshot_path}")
        except Exception as e:
            print(f"Could not save debug screenshot: {e}")
        
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
    
    def fill_search_form(self, driver: webdriver.Chrome, from_date: str, to_date: str, max_retries: int = 5) -> bool:
        """Fill and submit the search form with enhanced captcha handling
        
        Args:
            driver: Selenium WebDriver instance
            from_date: Start date in DD/MM/YYYY format
            to_date: End date in DD/MM/YYYY format
            max_retries: Maximum number of form submission attempts
            
        Returns:
            bool: True if form submission was successful, False otherwise
        """
        try:
            print(f"Filling search form: {from_date} to {to_date}")
            
            # Create debug directory
            debug_dir = os.path.join(os.getcwd(), "debug_screenshots")
            os.makedirs(debug_dir, exist_ok=True)
            
            wait = WebDriverWait(driver, 10)
            
            # Find date fields
            from_date_field = None
            to_date_field = None
            
            # Try multiple ways to find date fields
            date_selectors = [
                ("partyFromDate", "partyToDate"),  # Based on your log output
                ("fromDate", "toDate"),
                ("from_date", "to_date"),
                ("dateFrom", "dateTo"),
                ("dtFrom", "dtTo"),
                ("startDate", "endDate")
            ]
            
            for from_id, to_id in date_selectors:
                try:
                    from_date_field = driver.find_element(By.ID, from_id)
                    to_date_field = driver.find_element(By.ID, to_id)
                    print(f"Found date fields: {from_id}, {to_id}")
                    break
                except:
                    continue
            
            # If ID search failed, try by name
            if not from_date_field or not to_date_field:
                for from_id, to_id in date_selectors:
                    try:
                        from_date_field = driver.find_element(By.NAME, from_id)
                        to_date_field = driver.find_element(By.NAME, to_id)
                        print(f"Found date fields by name: {from_id}, {to_id}")
                        break
                    except:
                        continue
            
            # If still not found, try by XPath
            if not from_date_field or not to_date_field:
                xpath_selectors = [
                    "//input[contains(@id, 'from') or contains(@id, 'start') or contains(@id, 'begin')]",
                    "//input[contains(@id, 'to') or contains(@id, 'end')]"
                ]
                try:
                    from_elements = driver.find_elements(By.XPATH, xpath_selectors[0])
                    to_elements = driver.find_elements(By.XPATH, xpath_selectors[1])
                    if from_elements and to_elements:
                        from_date_field = from_elements[0]
                        to_date_field = to_elements[0]
                        print("Found date fields by XPath")
                except:
                    pass
            
            if not from_date_field or not to_date_field:
                print("Could not find date fields, taking debug screenshot")
                timestamp = int(time.time())
                screenshot_path = os.path.join(debug_dir, f"no_date_fields_{timestamp}.png")
                driver.save_screenshot(screenshot_path)
                return False
            
            # Clear and fill date fields
            print("Filling date fields...")
            from_date_field.clear()
            from_date_field.send_keys(from_date)
            print(f"Set from_date to {from_date}")
            
            to_date_field.clear()
            to_date_field.send_keys(to_date)
            print(f"Set to_date to {to_date}")
            
            # Set Reportable Judgment to YES
            try:
                # Look for reportable judgment radio buttons or dropdown
                reportable_yes_selectors = [
                    "//input[@id='rpjudgeY']",
                    "//input[@name='rpjudge' and @value='Y']",
                    "//input[contains(@id, 'reportable') and @value='Y']",
                    "//input[contains(@id, 'report') and @value='YES']",
                    "//select[@id='rptable']",
                    "//select[contains(@id, 'reportable')]",
                    "//select[contains(@name, 'reportable')]"
                ]
                
                for selector in reportable_yes_selectors:
                    try:
                        elements = driver.find_elements(By.XPATH, selector)
                        if elements:
                            element = elements[0]
                            if element.tag_name.lower() == "select":
                                # It's a dropdown
                                select = Select(element)
                                select.select_by_visible_text("YES")
                                print("Set Reportable Judgment dropdown to YES")
                            else:
                                # It's a radio button
                                element.click()
                                print("Clicked Reportable Judgment YES radio button")
                            break
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Could not set Reportable Judgment: {e}")
            
            # Handle captcha and submit with improved reliability
            for attempt in range(max_retries):
                try:
                    print(f"\nCaptcha attempt {attempt + 1}/{max_retries}")
                    
                    # Take screenshot before captcha solving
                    timestamp = int(time.time())
                    pre_captcha_path = os.path.join(debug_dir, f"pre_captcha_{timestamp}.png")
                    driver.save_screenshot(pre_captcha_path)
                    print(f"Saved pre-captcha screenshot to {pre_captcha_path}")
                    
                    # Get captcha text with increased attempts for automatic solving
                    captcha_text = self.solve_captcha(driver, max_attempts=5, attempt=attempt)
                    
                    if not captcha_text:
                        print("No captcha text provided")
                        if attempt < max_retries - 1:
                            print("Refreshing page...")
                            driver.refresh()
                            time.sleep(3)
                            if not self.navigate_to_search_form(driver):
                                return False
                            # Re-fill the form
                            from_date_field = driver.find_element(By.ID, from_id)
                            from_date_field.clear()
                            from_date_field.send_keys(from_date)
                            
                            to_date_field = driver.find_element(By.ID, to_id)
                            to_date_field.clear()
                            to_date_field.send_keys(to_date)
                            
                            # Set reportable to YES again
                            try:
                                if element.tag_name.lower() == "select":
                                    select = Select(element)
                                    select.select_by_visible_text("YES")
                                else:
                                    element.click()
                            except:
                                pass
                            continue
                        else:
                            return False
                    
                    # Find captcha input field with expanded selectors
                    captcha_field = None
                    captcha_id_selectors = ["txtCaptcha", "captcha", "Captcha", "CaptchaInput", "captchaInput", 
                                           "captchacode", "captcha_code", "security_code", "verification_code"]
                    
                    # Try by ID first
                    for selector in captcha_id_selectors:
                        try:
                            captcha_field = driver.find_element(By.ID, selector)
                            print(f"Found captcha field by ID: {selector}")
                            break
                        except:
                            continue
                    
                    # Try by name if ID failed
                    if not captcha_field:
                        for selector in captcha_id_selectors:
                            try:
                                captcha_field = driver.find_element(By.NAME, selector)
                                print(f"Found captcha field by name: {selector}")
                                break
                            except:
                                continue
                    
                    # Try by XPath as last resort
                    if not captcha_field:
                        xpath_selectors = [
                            "//input[contains(@id, 'captcha') or contains(@id, 'Captcha')]",
                            "//input[contains(@name, 'captcha') or contains(@name, 'Captcha')]",
                            "//input[contains(@placeholder, 'captcha') or contains(@placeholder, 'Captcha') or contains(@placeholder, 'enter text')]",
                            "//input[contains(@class, 'captcha') or contains(@class, 'Captcha')]",
                            "//input[following::img[contains(@src, 'captcha') or contains(@src, 'Captcha')]]",
                            "//input[preceding::img[contains(@src, 'captcha') or contains(@src, 'Captcha')]]",
                            # Try to find any input field near a captcha image
                            "//img[contains(@src, 'captcha')]/following::input[1]",
                            "//img[contains(@src, 'captcha')]/preceding::input[1]",
                            "//img[contains(@src, 'Captcha')]/following::input[1]",
                            "//img[contains(@src, 'Captcha')]/preceding::input[1]",
                            # Last resort - try to find any input field that might be a captcha field
                            "//input[@type='text' and (@maxlength='5' or @maxlength='6' or @maxlength='7' or @maxlength='8')]"
                        ]
                        
                        for selector in xpath_selectors:
                            try:
                                elements = driver.find_elements(By.XPATH, selector)
                                if elements:
                                    captcha_field = elements[0]
                                    print(f"Found captcha field by XPath: {selector}")
                                    break
                            except:
                                continue
                    
                    if not captcha_field:
                        print("Could not find captcha input field, taking debug screenshot")
                        screenshot_path = os.path.join(debug_dir, f"no_captcha_input_{timestamp}.png")
                        driver.save_screenshot(screenshot_path)
                        
                        if attempt < max_retries - 1:
                            continue
                        return False
                    
                    # Enter captcha with retry for input issues
                    max_input_attempts = 3
                    input_success = False
                    
                    for input_attempt in range(max_input_attempts):
                        try:
                            captcha_field.clear()
                            time.sleep(0.2)  # Small pause after clearing
                            captcha_field.send_keys(captcha_text)
                            time.sleep(0.2)  # Small pause after typing
                            
                            # Verify text was entered correctly
                            entered_text = captcha_field.get_attribute('value')
                            if entered_text == captcha_text:
                                input_success = True
                                print(f"Entered captcha text: {captcha_text}")
                                break
                            else:
                                print(f"Captcha text mismatch. Expected: {captcha_text}, Got: {entered_text}")
                        except Exception as e:
                            print(f"Error entering captcha text (attempt {input_attempt+1}): {e}")
                    
                    if not input_success:
                        print("Failed to enter captcha text after multiple attempts")
                        continue
                    
                    # Find and click submit button with expanded selectors
                    submit_button = None
                    submit_selectors = [
                        "//input[@type='submit']",
                        "//button[@type='submit']",
                        "//input[@value='Search']",
                        "//input[@value='Submit']",
                        "//input[@value='Go']",
                        "//input[contains(@value, 'Search')]",
                        "//input[contains(@value, 'submit')]",
                        "//input[contains(@value, 'Submit')]",
                        "//button[contains(text(), 'Search')]",
                        "//button[contains(text(), 'Submit')]",
                        "//button[contains(text(), 'Go')]",
                        "//button[contains(@class, 'search')]",
                        "//button[contains(@class, 'submit')]",
                        "//a[contains(@class, 'search')]",
                        "//a[contains(@class, 'submit')]",
                        "//a[contains(text(), 'Search')]",
                        "//a[contains(text(), 'Submit')]",
                        "//input[contains(@onclick, 'submit')]",
                        "//button[contains(@onclick, 'submit')]",
                        "//a[contains(@onclick, 'submit')]",
                        "//a[contains(@href, 'javascript') and contains(@href, 'submit')]",
                        # Last resort - try to find any button after the captcha input
                        "//input[contains(@id, 'captcha')]/following::input[@type='submit'][1]",
                        "//input[contains(@id, 'captcha')]/following::button[1]"
                    ]
                    
                    for selector in submit_selectors:
                        try:
                            buttons = driver.find_elements(By.XPATH, selector)
                            if buttons:
                                submit_button = buttons[0]
                                print(f"Found submit button: {selector}")
                                break
                        except:
                            continue
                    
                    if not submit_button:
                        print("Could not find submit button, taking debug screenshot")
                        screenshot_path = os.path.join(debug_dir, f"no_submit_button_{timestamp}.png")
                        driver.save_screenshot(screenshot_path)
                        
                        if attempt < max_retries - 1:
                            continue
                        return False
                    
                    # Take screenshot before submission
                    pre_submit_path = os.path.join(debug_dir, f"pre_submit_{timestamp}.png")
                    driver.save_screenshot(pre_submit_path)
                    
                    # Submit the form with improved click reliability
                    if self.wait_and_click(driver, submit_button):
                        print("Form submitted")
                        # Wait for results with progressive checking
                        for i in range(10):  # Check every second for 10 seconds
                            time.sleep(1)
                            # Check if page has changed
                            if any(indicator in driver.page_source.lower() for indicator in 
                                  ["judgment", "case", "result", "record", "table", "list", "no records"]):
                                break
                        
                        # Check if submission was successful
                        page_source = driver.page_source.lower()
                        current_url = driver.current_url.lower()
                        
                        # Check for error indicators
                        if any(error in page_source for error in ["invalid captcha", "incorrect captcha", "wrong captcha", "captcha mismatch"]):
                            print("Invalid captcha - retrying...")
                            # Take screenshot for debugging
                            screenshot_path = os.path.join(os.getcwd(), "captcha_debug", f"invalid_captcha_{attempt}.png")
                            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                            driver.save_screenshot(screenshot_path)
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
                                print("No clear success indicator - checking for errors...")
                                # Take screenshot for debugging
                                screenshot_path = os.path.join(os.getcwd(), "captcha_debug", f"unclear_result_{attempt}.png")
                                os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                                driver.save_screenshot(screenshot_path)
                                
                                # Check if we're still on the form page
                                if any(form_indicator in page_source for form_indicator in ["captcha", "search form", "parameter search"]):
                                    print("Still on form page - captcha likely failed")
                                    continue
                                else:
                                    print("Page changed but no clear indicators - assuming success")
                                    return True
                    else:
                        print("Failed to click submit button")
                        
                except Exception as e:
                    print(f"Error in attempt {attempt + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Refresh for retry
                if attempt < max_retries - 1:
                    print("Refreshing page for retry...")
                    driver.refresh()
                    time.sleep(3)
                    if not self.navigate_to_search_form(driver):
                        return False
            
            print("All submission attempts failed")
            return False
            
        except Exception as e:
            print(f"Error filling search form: {e}")
            return False
    
    def extract_judgment_data(self, driver: webdriver.Chrome) -> List[Dict]:
        """Extract judgment data from results with optimized performance"""
        judgments = []
        start_time = time.time()
        
        try:
            self.logger.info("Extracting judgment data...")
            
            # Use shorter wait time with explicit wait instead of sleep
            try:
                WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except TimeoutException:
                self.logger.warning("No tables found after waiting, continuing anyway")
            
            # Look for results table
            tables = driver.find_elements(By.TAG_NAME, "table")
            self.logger.info(f"Found {len(tables)} tables on page")
            
            results_table = None
            # Use parallel processing to analyze tables if there are many
            if len(tables) > 5:
                # Define a function to check if a table contains judgment data
                def is_judgment_table(table_idx):
                    table = tables[table_idx]
                    table_text = table.text.lower()
                    return any(keyword in table_text for keyword in ["s.no", "case", "judgment", "date", "party", "serial"]), table_idx
                
                # Process tables in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tables), 5)) as executor:
                    futures = [executor.submit(is_judgment_table, i) for i in range(len(tables))]
                    for future in concurrent.futures.as_completed(futures):
                        is_result_table, idx = future.result()
                        if is_result_table:
                            results_table = tables[idx]
                            self.logger.info(f"Selected table {idx+1} as results table")
                            break
            else:
                # Process sequentially for small number of tables
                for i, table in enumerate(tables):
                    table_text = table.text.lower()
                    self.logger.debug(f"Table {i+1} preview: {table_text[:100]}...")
                    
                    # Look for table with judgment data
                    if any(keyword in table_text for keyword in ["s.no", "case", "judgment", "date", "party", "serial"]):
                        results_table = table
                        self.logger.info(f"Selected table {i+1} as results table")
                        break
            
            if not results_table:
                self.logger.warning("No results table found")
                # Try to find any structured data
                rows = driver.find_elements(By.XPATH, "//tr[position()>1 and count(td)>2]")
                if rows:
                    self.logger.info(f"Found {len(rows)} data rows without clear table structure")
                    
                    # Define a function to process a row without table structure
                    def process_unstructured_row(row_idx):
                        try:
                            row = rows[row_idx]
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
                                
                                return judgment_data
                            return None
                        except Exception as e:
                            self.logger.error(f"Error processing unstructured row {row_idx}: {str(e)}")
                            return None
                    
                    # Process rows in parallel if there are many
                    if len(rows) > 10:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(rows), 10)) as executor:
                            futures = [executor.submit(process_unstructured_row, i) for i in range(len(rows))]
                            for future in concurrent.futures.as_completed(futures):
                                result = future.result()
                                if result:
                                    judgments.append(result)
                    else:
                        # Process sequentially for smaller sets
                        for i in range(len(rows)):
                            result = process_unstructured_row(i)
                            if result:
                                judgments.append(result)
                
                self.logger.info(f"Extracted {len(judgments)} judgments from unstructured data in {time.time() - start_time:.2f} seconds")
                return judgments
            
            # Extract headers with better error handling
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
                    
            except Exception as e:
                self.logger.warning(f"Error extracting headers: {str(e)}")
                headers = []
            
            if not headers:
                # Use default headers
                first_row = results_table.find_elements(By.TAG_NAME, "tr")[0] if results_table.find_elements(By.TAG_NAME, "tr") else None
                if first_row:
                    cell_count = len(first_row.find_elements(By.TAG_NAME, "td"))
                    headers = [f"Column_{i+1}" for i in range(cell_count)]
            
            self.logger.info(f"Table headers: {headers}")
            
            # Extract data rows
            rows = results_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
            self.logger.info(f"Processing {len(rows)} data rows...")
            
            # Define a function to process a single row
            def process_row(row_idx):
                try:
                    row = rows[row_idx]
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if not cells:
                        return None
                    
                    judgment_data = {}
                    for j, cell in enumerate(cells):
                        header_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                        cell_text = cell.text.strip()
                        judgment_data[header_name] = cell_text
                    
                    # Look for PDF download links with optimized selector
                    try:
                        pdf_links = row.find_elements(By.XPATH, ".//a[contains(@href, '.pdf') or contains(@href, 'download') or contains(text(), 'View') or contains(text(), 'Download')]")
                        if pdf_links:
                            judgment_data['pdf_url'] = pdf_links[0].get_attribute('href')
                            self.logger.debug(f"Found PDF link for row {row_idx+1}")
                        else:
                            judgment_data['pdf_url'] = ""
                    except Exception as pdf_err:
                        self.logger.warning(f"Error finding PDF link in row {row_idx+1}: {str(pdf_err)}")
                        judgment_data['pdf_url'] = ""
                    
                    # Only return if we have meaningful data
                    if any(value and str(value).strip() for value in judgment_data.values() if value != judgment_data.get('pdf_url', '')):
                        return judgment_data
                    return None
                        
                except Exception as e:
                    self.logger.error(f"Error processing row {row_idx+1}: {str(e)}")
                    return None
            
            # Process rows in parallel for better performance
            if len(rows) > 10:  # Use parallel processing for larger result sets
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(rows), 10)) as executor:
                    futures = [executor.submit(process_row, i) for i in range(len(rows))]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            judgments.append(result)
            else:  # Process sequentially for smaller result sets
                for i in range(len(rows)):
                    result = process_row(i)
                    if result:
                        judgments.append(result)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Successfully extracted {len(judgments)} judgment records in {elapsed_time:.2f} seconds")
            
            # Show sample of extracted data
            if judgments:
                self.logger.info("Sample judgment data:")
                sample = judgments[0]
                for key, value in sample.items():
                    if value:
                        self.logger.info(f"  {key}: {str(value)[:50]}...")
            
            # Calculate and log performance metrics
            if elapsed_time > 0 and len(judgments) > 0:
                records_per_second = len(judgments) / elapsed_time
                self.logger.info(f"Performance: {records_per_second:.2f} records/second")
            
            return judgments
            
        except Exception as e:
            self.logger.error(f"Error extracting judgment data: {str(e)}")
            # Log stack trace for debugging
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return judgments
    
    # Session cache for connection pooling
    _session = None
    
    def _get_session(self):
        """Get or create a requests session with proper headers for connection pooling"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*',
                'Referer': 'https://hcraj.nic.in/'
            })
        return self._session
        
    def download_pdf(self, url: str, filename: str) -> bool:
        """Download PDF from URL with optimized connection reuse"""
        start_time = time.time()
        try:
            self.logger.info(f"Downloading: {filename}")
            
            # Use persistent session for connection pooling
            session = self._get_session()
            
            # Use stream=True for memory efficiency and shorter timeout
            response = session.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # Ensure it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                self.logger.warning(f"URL may not be a PDF: {content_type}")
            
            pdf_path = self.pdf_dir / filename
            
            # Use larger chunk size for faster downloads
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):  # Increased chunk size
                    f.write(chunk)
            
            file_size = pdf_path.stat().st_size
            if file_size < 1024:  # Less than 1KB, likely an error page
                self.logger.warning(f"Downloaded file is too small ({file_size} bytes), likely an error")
                return False
            
            elapsed = time.time() - start_time
            self.logger.info(f"Successfully downloaded {filename} ({file_size} bytes) in {elapsed:.2f}s")
            return True
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout downloading PDF {filename}")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error downloading PDF {filename}")
            return False
        except Exception as e:
            self.logger.error(f"Error downloading PDF {filename}: {e}", exc_info=True)
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
        """Main scraping function with performance optimizations"""
        self.metrics["start_time"] = time.time()
        self.logger.info(f"Scraping judgments from {from_date} to {to_date}")
        
        try:
            driver = self.setup_driver()
        except Exception as e:
            self.logger.error(f"Failed to setup driver: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return []
        
        all_judgments = []
        
        try:
            # Navigate to search form
            if not self.navigate_to_search_form(driver):
                self.logger.error("Could not navigate to search form")
                self.metrics["errors"] += 1
                return []
            
            # Fill and submit form
            if not self.fill_search_form(driver, from_date, to_date):
                self.logger.error("Could not fill and submit form")
                self.metrics["errors"] += 1
                return []
            
            # Extract results
            judgments = self.extract_judgment_data(driver)
            
            if not judgments:
                self.logger.info("No judgments found")
                return []
            
            self.metrics["total_judgments"] = len(judgments)
            self.logger.info(f"Found {len(judgments)} judgments to process")
            
            # Process judgments in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Prepare arguments for parallel processing
                judgment_tasks = []
                for judgment in judgments:
                    judgment_id = self.generate_judgment_id(judgment)
                    
                    # Skip if already downloaded
                    if judgment_id in self.downloaded_judgments["downloaded_ids"]:
                        case_ref = list(judgment.values())[0] if judgment else "Unknown"
                        self.logger.debug(f"Skipping already downloaded: {case_ref[:50]}...")
                        continue
                    
                    judgment['judgment_id'] = judgment_id
                    judgment_tasks.append(judgment)
                
                # Submit tasks to thread pool
                futures = {executor.submit(self._process_judgment, judgment): judgment for judgment in judgment_tasks}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    judgment = futures[future]
                    try:
                        processed_judgment = future.result()
                        if processed_judgment:
                            all_judgments.append(processed_judgment)
                            # Mark as processed
                            self.downloaded_judgments["downloaded_ids"].add(processed_judgment['judgment_id'])
                            self.metrics["new_judgments"] += 1
                            if processed_judgment.get('download_status') == "Success":
                                self.metrics["download_success"] += 1
                            elif processed_judgment.get('download_status') == "Failed":
                                self.metrics["download_failed"] += 1
                    except Exception as e:
                        self.logger.error(f"Error processing judgment {judgment.get('judgment_id', 'unknown')}: {e}")
                        self.metrics["errors"] += 1
        
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}", exc_info=True)
            self.metrics["errors"] += 1
        
        finally:
            try:
                driver.quit()
            except:
                pass
            
            # Update metrics
            self.metrics["end_time"] = time.time()
            elapsed_time = self.metrics["end_time"] - self.metrics["start_time"]
            self.logger.info(f"Scraping completed in {elapsed_time:.2f} seconds. Found {len(all_judgments)} new judgments.")
            self.logger.info(f"Performance metrics: {self.metrics}")
        
        return all_judgments
        
    def _process_judgment(self, judgment: Dict) -> Dict:
        """Process a single judgment (for parallel execution)"""
        try:
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
            
            return judgment
                
        except Exception as e:
            self.logger.error(f"Error in _process_judgment: {e}", exc_info=True)
            return None
    
    def save_to_csv(self, judgments: List[Dict]):
        """Save judgments to CSV file with all relevant information using optimized parallel processing
        
        The CSV includes the following columns:
        - judgment_id: Unique identifier for the judgment
        - case_number: Case number from the court
        - petitioner: Name of the petitioner
        - respondent: Name of the respondent
        - judgment_date: Date of the judgment
        - judge_name: Name of the judge
        - pdf_url: URL to download the PDF
        - pdf_path: Local path to the downloaded PDF
        - pdf_filename: Filename of the downloaded PDF
        - download_status: Status of the download (success, failed, skipped)
        - scraped_date: Date and time when the judgment was scraped
        - file_size_kb: Size of the PDF file in KB
        """
        start_time = time.time()
        
        if not judgments:
            self.logger.info("No new judgments to save")
            return
        
        self.logger.info(f"Saving {len(judgments)} judgments to CSV")
        
        # Add file size information to new judgments in parallel
        def get_file_size(judgment):
            if judgment.get('pdf_filename') and not judgment.get('file_size_kb'):
                pdf_path = self.pdf_dir / judgment['pdf_filename']
                if pdf_path.exists():
                    try:
                        file_size_bytes = os.path.getsize(pdf_path)
                        judgment['file_size_kb'] = round(file_size_bytes / 1024, 2)
                    except Exception as e:
                        self.logger.warning(f"Error getting file size for {judgment.get('pdf_filename')}: {str(e)}")
                        judgment['file_size_kb'] = None
            return judgment
        
        # Process file sizes in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(judgments), 20)) as executor:
            processed_judgments = list(executor.map(get_file_size, judgments))
        
        # Load existing data with error handling
        existing_df = pd.DataFrame()
        if self.csv_file.exists():
            try:
                existing_df = pd.read_csv(self.csv_file)
                self.logger.info(f"Loaded {len(existing_df)} existing records from {self.csv_file}")
            except Exception as e:
                self.logger.warning(f"Could not load existing CSV: {str(e)}")
                # Create backup of potentially corrupted file
                if self.csv_file.exists():
                    backup_file = self.csv_file.with_suffix('.csv.bak')
                    try:
                        shutil.copy2(self.csv_file, backup_file)
                        self.logger.info(f"Created backup of CSV file at {backup_file}")
                    except Exception as backup_err:
                        self.logger.error(f"Failed to create backup: {str(backup_err)}")
        
        # Create new DataFrame
        new_df = pd.DataFrame(processed_judgments)
        
        # Combine data efficiently
        if not existing_df.empty:
            # Get all columns from both dataframes
            all_columns = set(existing_df.columns) | set(new_df.columns)
            
            # Fill missing columns with empty strings
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = ""
                if col not in new_df.columns:
                    new_df[col] = ""
            
            # Concatenate dataframes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates based on judgment_id if present
        if 'judgment_id' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['judgment_id'], keep='last')
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"CSV processing completed in {elapsed_time:.2f} seconds")
        
        # Ensure all required columns are present
        required_columns = [
            'judgment_id', 'case_number', 'case_title', 'judgment_date',
            'pdf_url', 'pdf_filename', 'download_status', 'scraped_date',
            'petitioner', 'respondent', 'judge_name', 'pdf_path', 'file_size_kb'
        ]
        
        # Initialize missing columns with empty strings
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = ""
        
        # Reorder columns to have a consistent format
        available_columns = [col for col in required_columns if col in combined_df.columns]
        other_columns = [col for col in combined_df.columns if col not in required_columns]
        combined_df = combined_df[available_columns + other_columns]
        
        try:
            # Save to CSV with optimized settings
            combined_df.to_csv(self.csv_file, index=False)
            self.logger.info(f"Successfully saved {len(combined_df)} records to {self.csv_file}")
            self.logger.info(f"Added {len(judgments)} new judgments to database")
            
            # Update statistics
            total_size_mb = combined_df['file_size_kb'].sum() / 1024 if 'file_size_kb' in combined_df.columns else 0
            self.logger.info(f"Total PDF size: {total_size_mb:.2f} MB")
            
            # Return success
            return True
        except Exception as e:
            self.logger.error(f"Error saving CSV file: {str(e)}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def _solve_with_advanced_ocr(self, driver, attempt=0) -> str:
        """Use the advanced OCR captcha solver"""
        try:
            # Create directory for captcha images
            captcha_dir = Path("captcha_images")
            captcha_dir.mkdir(exist_ok=True)
            
            # Generate unique filename for this attempt
            timestamp = int(time.time())
            captcha_filename = f"captcha_{timestamp}_{attempt}.png"
            captcha_path = captcha_dir / captcha_filename
            
            # Find captcha image using multiple strategies
            captcha_img = self._find_captcha_image(driver, attempt)
            
            if captcha_img:
                # Save the captcha image
                captcha_img.screenshot(str(captcha_path))
                self.logger.info(f"Saved captcha image to {captcha_path}")
                
                # Solve with advanced OCR
                captcha_text, confidence = self.captcha_solver.solve_captcha(str(captcha_path))
                
                if captcha_text and confidence > 0.5:
                    self.logger.info(f"Captcha solved: {captcha_text} (confidence: {confidence:.2f})")
                    self.metrics["captcha_success"] += 1
                    return captcha_text
                else:
                    self.logger.warning(f"Low confidence captcha solution: {captcha_text} ({confidence:.2f})")
            else:
                self.logger.warning("Could not find captcha image")
                
            return ""
        except Exception as e:
            self.logger.error(f"Error in advanced captcha solving: {e}", exc_info=True)
            return ""
    
    def _find_captcha_image(self, driver, attempt=0):
        """Find captcha image using multiple strategies"""
        # Multiple ways to find captcha image with expanded selectors
        captcha_selectors = [
            "//img[contains(@src, 'captcha')]",
            "//img[contains(@src, 'Captcha')]",
            "//img[contains(@src, 'CAPTCHA')]",
            "//img[contains(@id, 'captcha')]",
            "//img[contains(@id, 'Captcha')]",
            "//img[contains(@class, 'captcha')]",
            "//img[contains(@alt, 'captcha')]",
            "//img[contains(@title, 'captcha')]",
            "//img[contains(@name, 'captcha')]",
            "//img[contains(@name, 'Captcha')]",
            "//img[contains(@src, 'jcaptcha')]",
            "//img[contains(@src, 'securimage')]",
            "//img[contains(@src, 'random')]",
            "//img[contains(@src, 'verify')]",
            "//img[contains(@src, 'verification')]"
        ]
        
        # Rotate through different selector strategies based on attempt number
        if attempt > 0:
            # Reorder selectors to try different ones first on subsequent attempts
            captcha_selectors = captcha_selectors[attempt % len(captcha_selectors):] + captcha_selectors[:attempt % len(captcha_selectors)]
        
        # Try standard selectors first
        for selector in captcha_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    for i, img in enumerate(elements):
                        # On subsequent attempts, try different images if multiple match
                        if i == (attempt % max(1, len(elements))):
                            self.logger.info(f"Found captcha with selector: {selector} (image {i+1} of {len(elements)})")
                            return img
            except Exception as e:
                self.logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        # If standard selectors failed, try proximity-based detection
        try:
            # Try to find captcha input field first
            captcha_input = None
            input_selectors = [
                "//input[contains(@id, 'captcha') or contains(@id, 'Captcha')]",
                "//input[contains(@name, 'captcha') or contains(@name, 'Captcha')]",
                "//input[contains(@placeholder, 'captcha') or contains(@placeholder, 'enter text')]",
                "//input[contains(@class, 'captcha')]"
            ]
            
            for selector in input_selectors:
                inputs = driver.find_elements(By.XPATH, selector)
                if inputs:
                    captcha_input = inputs[0]
                    self.logger.info(f"Found captcha input with selector: {selector}")
                    break
            
            if captcha_input:
                # Look for images near the captcha input
                # Get the location of the input field
                input_location = captcha_input.location
                
                # Find all images on the page
                all_images = driver.find_elements(By.TAG_NAME, "img")
                
                # Sort images by proximity to the input field
                def distance(img):
                    try:
                        img_loc = img.location
                        return ((img_loc['x'] - input_location['x'])**2 + 
                                (img_loc['y'] - input_location['y'])**2)**0.5
                    except:
                        return 99999
                
                nearby_images = sorted(all_images, key=distance)
                
                # Use the closest image or the one corresponding to the attempt number
                if nearby_images:
                    img_index = min(attempt, len(nearby_images)-1)
                    self.logger.info(f"Using proximity-based image detection (image {img_index+1} of {len(nearby_images)})")
                    return nearby_images[img_index]
            else:
                # If we can't find the input field, try using small images as they're often captchas
                small_images = []
                for img in driver.find_elements(By.TAG_NAME, "img"):
                    try:
                        size = img.size
                        if 20 <= size['width'] <= 200 and 10 <= size['height'] <= 80:
                            small_images.append(img)
                    except:
                        continue
                
                if small_images:
                    img_index = attempt % len(small_images)
                    self.logger.info(f"Using size-based image detection (image {img_index+1} of {len(small_images)})")
                    return small_images[img_index]
        except Exception as e:
            self.logger.error(f"Error in proximity detection: {e}")
        
        return None
    
    def run_incremental_scrape(self, days_back: int = 10) -> List[Dict]:
        """Run the scraper with incremental download functionality
        
        Args:
            days_back: Number of days to look back from today
            
        Returns:
            List of judgment dictionaries
        """
        # Start performance tracking
        start_time = time.time()
        self.metrics["start_time"] = start_time
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Running incremental scrape from {start_date} to {end_date}")
        
        # Update last run date
        self.downloaded_judgments["last_run_date"] = end_date.isoformat()
        
        # Initialize WebDriver
        driver = None
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
            driver.set_page_load_timeout(30)  # Set page load timeout
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            self.metrics["errors"] += 1
            return []
        
        try:
            # Navigate to the website with enhanced retry logic
            max_page_load_attempts = 5  # Increased from 3 to 5
            page_loaded = False
            
            for attempt in range(max_page_load_attempts):
                try:
                    # Clear cookies and cache before loading
                    if attempt > 0:
                        driver.delete_all_cookies()
                        self.logger.info("Cleared cookies for fresh attempt")
                    
                    # Load the page with increased timeout
                    self.logger.info(f"Attempting to load page: {self.base_url} (attempt {attempt+1}/{max_page_load_attempts})")
                    driver.set_page_load_timeout(45)  # Increased timeout
                    driver.get(self.base_url)
                    
                    # Take screenshot for debugging
                    debug_path = os.path.join(os.getcwd(), "debug_screenshots")
                    os.makedirs(debug_path, exist_ok=True)
                    timestamp = int(time.time())
                    screenshot_path = os.path.join(debug_path, f"page_load_attempt_{attempt+1}_{timestamp}.png")
                    driver.save_screenshot(screenshot_path)
                    self.logger.info(f"Saved page load screenshot to {screenshot_path}")
                    
                    # Try multiple selectors to determine if page loaded
                    selectors_to_try = [(By.ID, "txtFromDate"), (By.ID, "txtToDate"), 
                                       (By.XPATH, "//input[@type='text']"), 
                                       (By.TAG_NAME, "body")]
                    
                    for selector in selectors_to_try:
                        try:
                            WebDriverWait(driver, 15).until(EC.presence_of_element_located(selector))
                            self.logger.info(f"Page loaded successfully (detected element: {selector})")
                            page_loaded = True
                            break
                        except Exception:
                            continue
                    
                    if page_loaded:
                        break
                        
                    self.logger.warning("Could not detect expected elements, but page might be partially loaded")
                    # Check if we have any content
                    if "Rajasthan High Court" in driver.page_source or len(driver.page_source) > 1000:
                        self.logger.info("Page has content, proceeding with caution")
                        page_loaded = True
                        break
                        
                except TimeoutException:
                    self.logger.warning(f"Timeout waiting for page to load (attempt {attempt+1}/{max_page_load_attempts})")
                    if attempt < max_page_load_attempts - 1:
                        self.logger.info("Retrying page load...")
                        try:
                            driver.execute_script("window.stop();")
                        except Exception:
                            pass
                        time.sleep(2)  # Wait before retry
                except WebDriverException as e:
                    self.logger.error(f"WebDriver error: {e}")
                    if "net::ERR_CONNECTION_TIMED_OUT" in str(e) and attempt < max_page_load_attempts - 1:
                        self.logger.info("Connection timed out, retrying...")
                        time.sleep(5)  # Longer wait for network issues
                        continue
                    break
            
            if not page_loaded:
                self.logger.error("Failed to load page after multiple attempts")
                self.metrics["errors"] += 1
                return []
            
            # Set date range with explicit waits and JavaScript for reliable date entry
            try:
                # Wait for date inputs to be interactive
                from_date_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "txtFromDate"))
                )
                
                # Use JavaScript to set date values directly - more reliable than send_keys
                formatted_start_date = start_date.strftime("%d/%m/%Y")
                driver.execute_script(f"document.getElementById('txtFromDate').value = '{formatted_start_date}'")
                self.logger.info(f"Set from date to {formatted_start_date}")
                
                to_date_input = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "txtToDate"))
                )
                
                formatted_end_date = end_date.strftime("%d/%m/%Y")
                driver.execute_script(f"document.getElementById('txtToDate').value = '{formatted_end_date}'")
                self.logger.info(f"Set to date to {formatted_end_date}")
                
                # Verify date fields were set correctly
                actual_from_date = driver.execute_script("return document.getElementById('txtFromDate').value")
                actual_to_date = driver.execute_script("return document.getElementById('txtToDate').value")
                
                if actual_from_date != formatted_start_date or actual_to_date != formatted_end_date:
                    self.logger.warning(f"Date verification failed. Expected: {formatted_start_date}-{formatted_end_date}, Got: {actual_from_date}-{actual_to_date}")
                    # Retry with traditional method if JavaScript approach failed
                    from_date_input.clear()
                    from_date_input.send_keys(formatted_start_date)
                    to_date_input.clear()
                    to_date_input.send_keys(formatted_end_date)
                
                # Set Reportable Judgment to YES
                reportable_select = Select(WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "ddlReportable"))
                ))
                reportable_select.select_by_visible_text("YES")
            except NoSuchElementException as e:
                self.logger.error(f"Element not found: {e}")
                self.metrics["errors"] += 1
                return []
            except TimeoutException as e:
                self.logger.error(f"Timeout waiting for form elements: {e}")
                self.metrics["errors"] += 1
                return []
            
            # Solve captcha with enhanced handling
            captcha_solved = False
            max_captcha_attempts = 5
            captcha_attempt = 0
            
            while not captcha_solved and captcha_attempt < max_captcha_attempts:
                self.logger.info(f"Captcha attempt {captcha_attempt + 1}/{max_captcha_attempts}")
                
                # Take screenshot of captcha for debugging
                try:
                    captcha_img = driver.find_element(By.ID, "imgCaptcha")
                    if captcha_img:
                        timestamp = int(time.time())
                        debug_path = os.path.join(os.getcwd(), "captcha_debug")
                        os.makedirs(debug_path, exist_ok=True)
                        screenshot_path = os.path.join(debug_path, f"captcha_attempt_{captcha_attempt+1}_{timestamp}.png")
                        captcha_img.screenshot(screenshot_path)
                        self.logger.info(f"Saved captcha screenshot to {screenshot_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save captcha screenshot: {e}")
                
                # Try OCR first
                captcha_text = self.solve_captcha_ocr(driver, captcha_attempt)
                
                if not captcha_text:
                    # If OCR fails, try manual input
                    self.logger.info("Captcha OCR failed. Please solve the captcha manually:")
                    captcha_text = input("Enter captcha text: ")
                
                # Ensure captcha text is properly formatted
                if captcha_text:
                    # Remove any whitespace and convert to uppercase (common captcha format)
                    captcha_text = captcha_text.strip().upper()
                    self.logger.info(f"Attempting captcha with text: {captcha_text}")
                
                # Enter captcha with retry logic
                max_input_attempts = 3
                input_success = False
                
                for input_attempt in range(max_input_attempts):
                    try:
                        # Wait for captcha input field to be ready
                        captcha_input = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "txtCaptcha"))
                        )
                        
                        # Clear field and enter text
                        captcha_input.clear()
                        time.sleep(0.5)  # Small pause to ensure field is clear
                        
                        # Try both methods to ensure text entry works
                        captcha_input.send_keys(captcha_text)
                        
                        # Verify text was entered correctly
                        entered_text = captcha_input.get_attribute('value')
                        if entered_text != captcha_text:
                            self.logger.warning(f"Captcha text verification failed. Expected: {captcha_text}, Got: {entered_text}")
                            # Try JavaScript as fallback
                            driver.execute_script(f"document.getElementById('txtCaptcha').value = '{captcha_text}'")
                        else:
                            input_success = True
                            break
                    except Exception as e:
                        self.logger.error(f"Error entering captcha (attempt {input_attempt+1}): {e}")
                        time.sleep(1)
                
                if not input_success:
                    self.logger.error("Failed to enter captcha text after multiple attempts")
                    captcha_attempt += 1
                    continue
                
                # Click search button with retry logic
                try:
                    search_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.ID, "btnSearch"))
                    )
                    # Use JavaScript click for reliability
                    driver.execute_script("arguments[0].click();", search_button)
                    self.logger.info("Clicked search button")
                    
                    # Wait for results or error message with explicit wait
                    WebDriverWait(driver, 10).until(
                        lambda d: len(d.find_elements(By.XPATH, "//div[contains(text(), 'Invalid Captcha')]")) > 0 or 
                                len(d.find_elements(By.ID, "grdJudgement")) > 0 or
                                len(d.find_elements(By.XPATH, "//div[contains(text(), 'No Record Found')]")) > 0
                    )
                except TimeoutException:
                    self.logger.warning("Timeout waiting for search results or error message")
                    # Take screenshot for debugging
                    debug_path = os.path.join(os.getcwd(), "debug_screenshots")
                    os.makedirs(debug_path, exist_ok=True)
                    timestamp = int(time.time())
                    screenshot_path = os.path.join(debug_path, f"search_timeout_{timestamp}.png")
                    driver.save_screenshot(screenshot_path)
                    self.logger.info(f"Saved debug screenshot to {screenshot_path}")
                    captcha_attempt += 1
                    continue
                except Exception as e:
                    self.logger.error(f"Error during search: {e}")
                    captcha_attempt += 1
                    continue
                
                # Check if captcha was successful using multiple detection methods
                error_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Invalid Captcha') or contains(text(), 'incorrect captcha')]")
                error_messages = driver.find_elements(By.XPATH, "//span[contains(text(), 'Invalid Captcha') or contains(text(), 'incorrect captcha')]")
                
                if not error_elements and not error_messages:
                    # Check if results table or no records message is present
                    results_table = driver.find_elements(By.ID, "grdJudgement")
                    no_records = driver.find_elements(By.XPATH, "//div[contains(text(), 'No Record Found')]")
                    
                    if results_table or no_records:
                        captcha_solved = True
                        self.logger.info(f"Captcha solved successfully: {captcha_text}")
                        self.metrics["captcha_success"] += 1
                    else:
                        self.logger.warning("No error message but also no results table found")
                        captcha_attempt += 1
                else:
                    error_text = error_elements[0].text if error_elements else (error_messages[0].text if error_messages else "Unknown error")
                    self.logger.warning(f"Captcha attempt {captcha_attempt + 1} failed: {error_text}")
                    captcha_attempt += 1
                    self.metrics["captcha_attempts"] += 1
                    
                    # Refresh captcha with multiple selector attempts
                    refresh_success = False
                    for selector_id in ["btnRefresh", "refresh", "captchaRefresh"]:
                        try:
                            refresh_button = WebDriverWait(driver, 3).until(
                                EC.element_to_be_clickable((By.ID, selector_id))
                            )
                            refresh_button.click()
                            self.logger.info(f"Refreshed captcha using selector: {selector_id}")
                            refresh_success = True
                            time.sleep(1)  # Wait for captcha to refresh
                            break
                        except Exception:
                            continue
                    
                    if not refresh_success:
                        # Try XPath selectors as fallback
                        for xpath in ["//img[contains(@src, 'refresh')]", "//a[contains(@onclick, 'captcha')]"]:
                            try:
                                refresh_elements = driver.find_elements(By.XPATH, xpath)
                                if refresh_elements:
                                    refresh_elements[0].click()
                                    self.logger.info(f"Refreshed captcha using XPath: {xpath}")
                                    time.sleep(1)  # Wait for captcha to refresh
                                    break
                            except Exception:
                                continue
            
            if not captcha_solved:
                self.logger.error("Failed to solve captcha after multiple attempts. Exiting.")
                self.metrics["errors"] += 1
                return []
            
            # Wait for results table
            try:
                # Use shorter timeout with more specific condition
                WebDriverWait(driver, 15).until(
                    lambda d: len(d.find_elements(By.ID, "grdJudgement")) > 0 or 
                              len(d.find_elements(By.XPATH, "//div[contains(text(), 'No Record Found')]")) > 0
                )
            except TimeoutException:
                self.logger.warning("No results found or table not loaded.")
                # Check if no records found
                no_records_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'No Record Found')]")
                if no_records_elements:
                    self.logger.info("No records found for the specified date range.")
                return []
            
            # Extract judgments
            judgments = self.extract_judgments(driver)
            
            # Save to CSV if we have judgments
            if judgments:
                self.save_to_csv(judgments)
            else:
                self.logger.info("No judgments extracted, skipping CSV save")
            
            # Save state
            self.save_state()
            
            # Update metrics
            end_time = time.time()
            self.metrics["end_time"] = end_time
            self.metrics["total_judgments"] = len(judgments)
            
            # Log performance metrics
            duration = end_time - start_time
            judgments_per_second = len(judgments) / duration if duration > 0 else 0
            self.logger.info(f"Scraping completed in {duration:.2f} seconds ({judgments_per_second:.2f} judgments/sec)")
            self.logger.info(f"Total judgments: {self.metrics['total_judgments']}")
            self.logger.info(f"New judgments: {self.metrics['new_judgments']}")
            self.logger.info(f"Captcha success rate: {self.metrics['captcha_success']}/{self.metrics['captcha_attempts']}")
            self.logger.info(f"Download success rate: {self.metrics['download_success']}/{self.metrics['download_success'] + self.metrics['download_failed']}")
            
            return judgments
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            self.metrics["errors"] += 1
            return []
        finally:
            # Close the browser
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    self.logger.error(f"Error closing browser: {e}")
                    import traceback
                    self.logger.debug(f"Stack trace: {traceback.format_exc()}")

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
                