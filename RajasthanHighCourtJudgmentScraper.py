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
        if not TESSERACT_AVAILABLE:
            print("Tesseract OCR not available")
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
                    except Exception as e:
                        print(f"Error refreshing captcha: {e}")
                        # Look for captcha refresh button with multiple selectors
                except Exception as e:
                    print(f"Error taking screenshot: {e}")
                    return ""
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
                            refresh_selectors = [
                                "//img[contains(@src, 'refresh')]",
                                "//img[contains(@title, 'refresh')]",
                                "//img[contains(@alt, 'refresh')]",
                                "//a[contains(@onclick, 'captcha')]",
                                "//a[contains(@href, 'javascript') and contains(@href, 'captcha')]"
                            ]
                            
                            for selector in refresh_selectors:
                                refresh_buttons = driver.find_elements(By.XPATH, selector)
                                if refresh_buttons:
                                    refresh_buttons[0].click()
                                    print(f"Refreshed captcha on attempt {attempt+1}")
                                    time.sleep(1)  # Wait for new captcha to load
                                    break
                        except Exception as e:
                            print(f"Could not refresh captcha: {e}")
                    
                    # Try OCR with the current attempt number
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
    
    def navigate_to_search_form(self, driver webdriver.Chrome) -> bool:
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
        """Save judgments to CSV file with all relevant information
        
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
        if not judgments:
            print("No new judgments to save")
            return
        
        # Add file size information to new judgments
        for judgment in judgments:
            # Add file size information if PDF exists
            if judgment.get('pdf_path') and os.path.exists(judgment['pdf_path']):
                try:
                    file_size_bytes = os.path.getsize(judgment['pdf_path'])
                    judgment['file_size_kb'] = round(file_size_bytes / 1024, 2)
                except Exception as e:
                    print(f"Error getting file size: {e}")
                    judgment['file_size_kb'] = None
        
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
        
        # Ensure all required columns are present
        required_columns = [
            'judgment_id', 'case_number', 'petitioner', 'respondent', 
            'judgment_date', 'judge_name', 'pdf_url', 'pdf_path', 'pdf_filename',
            'download_status', 'scraped_date', 'file_size_kb'
        ]
        
        # Initialize missing columns with None
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = None
        
        # Reorder columns to have a consistent format
        available_columns = [col for col in required_columns if col in combined_df.columns]
        other_columns = [col for col in combined_df.columns if col not in required_columns]
        combined_df = combined_df[available_columns + other_columns]
        
        # Save to CSV
        combined_df.to_csv(self.csv_file, index=False)
        print(f"Saved {len(judgments)} new judgments to {self.csv_file}")
        print(f"Total judgments in database: {len(combined_df)}")
    
    def run_incremental_scrape(self, days_back: int = 10):
        """Run incremental scraping
        
        This function implements the incremental daily download functionality as specified:
        1. First run: Downloads all judgments from (today - days_back) to today
        2. Subsequent runs: Downloads only new judgments that weren't downloaded before
           - This includes judgments from the new day
           - This also includes any judgments from previous days that were uploaded after our last run
        """
        today = datetime.now()
        from_date_obj = today - timedelta(days=days_back)
        
        from_date = from_date_obj.strftime("%d/%m/%Y")
        to_date = today.strftime("%d/%m/%Y")
        
        print(f"Running incremental scrape from {from_date} to {to_date}")
        print(f"Looking for judgments in the last {days_back} days")
        
        # Check if this is a subsequent run and we need to adjust the from_date
        if self.downloaded_judgments.get("last_run_date"):
            try:
                # Calculate the date range for this run
                last_run_date = datetime.fromisoformat(self.downloaded_judgments["last_run_date"])
                print(f"Last run was on: {last_run_date.strftime('%d/%m/%Y')}")
                
                # For subsequent runs, we still need to check the full date range
                # This ensures we catch any judgments that were added to the website after our last run
                # but with earlier dates
                print("This is a subsequent run - will check for new judgments in the entire date range")
                
                # We'll still use the same date range, but we'll skip judgments we've already downloaded
                # based on their unique ID in the scrape_judgments function
            except Exception as e:
                print(f"Error parsing last run date: {e}")
        else:
            print("This is the first run - will download all judgments in the date range")
        
        judgments = self.scrape_judgments(from_date, to_date)
        
        if judgments:
            self.save_to_csv(judgments)
            print(f"Downloaded {len(judgments)} new judgments")
        else:
            print("No new judgments found")
        
        # Update state
        self.downloaded_judgments["last_run_date"] = today.isoformat()
        self.save_state()
        
        print(f"\nScraping completed!")
        print(f"- Total judgments downloaded in this run: {len(judgments)}")
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
                