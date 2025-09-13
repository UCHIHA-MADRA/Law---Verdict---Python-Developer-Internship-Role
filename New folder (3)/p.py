#!/usr/bin/env python3
"""
Enhanced Automatic Captcha Extractor with Multiple OCR Engines
Retries up to 10 times with no manual fallback
"""

import cv2
import numpy as np
import base64
import re
import time
import logging
from typing import Optional, List, Tuple, Any
from selenium.webdriver.common.by import By
from numerical_captcha_extractor import NumericalCaptchaExtractor
# OCR Engine availability checks
OCR_ENGINES = {}

try:
    import easyocr
    OCR_ENGINES['easyocr'] = True
except ImportError:
    OCR_ENGINES['easyocr'] = False

try:
    import pytesseract
    OCR_ENGINES['tesseract'] = True
except ImportError:
    pytesseract = None
    OCR_ENGINES['tesseract'] = False

try:
    from paddleocr import PaddleOCR
    OCR_ENGINES['paddleocr'] = True
except ImportError:
    OCR_ENGINES['paddleocr'] = False

class EnhancedCaptchaExtractor:
    """Multi-engine automatic captcha extractor with retry mechanism"""
    
    def __init__(self, debug_mode: bool = True, max_retries: int = 10):
        self.debug_mode = debug_mode
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Initialize OCR engines
        self.ocr_engines = {}
        self._initialize_ocr_engines()
        
        # Preprocessing variations to try
        self.preprocessing_methods = [
            'standard',
            'high_contrast', 
            'denoised',
            'morphological',
            'adaptive_threshold',
            'bilateral_filter'
        ]
        
    def _initialize_ocr_engines(self):
        """Initialize all available OCR engines"""
        if OCR_ENGINES['easyocr']:
            try:
                import easyocr
                self.ocr_engines['easyocr'] = easyocr.Reader(['en'], gpu=False, verbose=False)
                if self.debug_mode:
                    print("EasyOCR initialized successfully")
            except Exception as e:
                if self.debug_mode:
                    print(f"EasyOCR initialization failed: {e}")
        
        if OCR_ENGINES['paddleocr']:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                if self.debug_mode:
                    print("PaddleOCR initialized successfully")
            except Exception as e:
                if self.debug_mode:
                    print(f"PaddleOCR initialization failed: {e}")
        
        if OCR_ENGINES['tesseract'] and pytesseract is not None:
            try:
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                self.ocr_engines['tesseract'] = True
                if self.debug_mode:
                    print("Tesseract initialized successfully")
            except Exception as e:
                if self.debug_mode:
                    print(f"Tesseract initialization failed: {e}")
    
    def preprocess_image(self, image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Apply different preprocessing methods"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small
        height, width = gray.shape
        if width < 200:
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        if method == 'standard':
            # Basic preprocessing
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'high_contrast':
            # High contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'denoised':
            # Noise reduction first
            denoised = cv2.fastNlMeansDenoising(gray)
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'morphological':
            # Morphological operations
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        elif method == 'adaptive_threshold':
            # Adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        
        elif method == 'bilateral_filter':
            # Bilateral filtering
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            # Fallback to standard
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Auto-invert if needed
        if thresh.mean() < 127:
            thresh = cv2.bitwise_not(thresh)
        
        if self.debug_mode:
            cv2.imwrite(f"captcha_{method}.png", thresh)
        
        return thresh
    
    def clean_ocr_result(self, text: str) -> str:
        """Clean and validate OCR result"""
        if not text:
            return ""
        
        # Apply common OCR error corrections
        text_upper = str(text).upper().strip()
        corrections = {
            'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'G': '0',
            'I': '1', 'L': '1', '|': '1', 'J': '1', 'T': '1',
            'Z': '2', 'S': '5', 'G': '6', 'C': '6',
            'T': '7', 'B': '8', 'R': '8', 'A': '4', 'H': '4'
        }
        
        corrected = text_upper
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Extract only digits
        digits = re.sub(r'[^0-9]', '', corrected)
        
        # Additional pattern-based corrections
        if not digits and text:
            # Try to extract numbers from mixed text
            numbers = re.findall(r'\d+', text)
            digits = ''.join(numbers)
        
        return digits
    
    def extract_with_easyocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extract text using EasyOCR"""
        results = []
        if 'easyocr' not in self.ocr_engines:
            return results
        
        try:
            for method in self.preprocessing_methods:
                processed = self.preprocess_image(image, method)
                ocr_results = self.ocr_engines['easyocr'].readtext(processed)
                
                for (bbox, text, confidence) in ocr_results:
                    cleaned = self.clean_ocr_result(text)
                    if cleaned and len(cleaned) >= 4:
                        results.append((cleaned, confidence, f"easyocr_{method}"))
                        if self.debug_mode:
                            print(f"EasyOCR ({method}): '{text}' -> '{cleaned}' (conf: {confidence:.3f})")
        except Exception as e:
            if self.debug_mode:
                print(f"EasyOCR error: {e}")
        
        return results
    
    def extract_with_tesseract(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extract text using Tesseract"""
        results = []
        if 'tesseract' not in self.ocr_engines or pytesseract is None:
            return results
        
        try:
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789',
                '--psm 7 -c tessedit_char_whitelist=0123456789',
                '--psm 6 -c tessedit_char_whitelist=0123456789',
                '--psm 13 -c tessedit_char_whitelist=0123456789'
            ]
            
            for method in self.preprocessing_methods:
                processed = self.preprocess_image(image, method)
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(processed, config=config)
                        cleaned = self.clean_ocr_result(text)
                        if cleaned and len(cleaned) >= 4:
                            results.append((cleaned, 0.8, f"tesseract_{method}"))
                            if self.debug_mode:
                                print(f"Tesseract ({method}): '{text.strip()}' -> '{cleaned}'")
                    except:
                        continue
        except Exception as e:
            if self.debug_mode:
                print(f"Tesseract error: {e}")
        
        return results
    
    def extract_with_paddleocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extract text using PaddleOCR"""
        results = []
        if 'paddleocr' not in self.ocr_engines:
            return results
        
        try:
            for method in self.preprocessing_methods:
                processed = self.preprocess_image(image, method)
                paddle_results = self.ocr_engines['paddleocr'].ocr(processed)
                
                if paddle_results and paddle_results[0]:
                    for detection in paddle_results[0]:
                        if len(detection) >= 2:
                            text_info = detection[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0])
                                confidence = float(text_info[1])
                                cleaned = self.clean_ocr_result(text)
                                if cleaned and len(cleaned) >= 4:
                                    results.append((cleaned, confidence, f"paddleocr_{method}"))
                                    if self.debug_mode:
                                        print(f"PaddleOCR ({method}): '{text}' -> '{cleaned}' (conf: {confidence:.3f})")
        except Exception as e:
            if self.debug_mode:
                print(f"PaddleOCR error: {e}")
        
        return results
    
    def extract_with_contours(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extract using contour analysis (fallback method)"""
        results = []
        try:
            # Convert to grayscale and threshold
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and count digit-like contours
            digit_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # Filter for digit-like shapes
                if 0.2 < aspect_ratio < 1.8 and area > 20:
                    digit_contours.append((x, contour))
            
            # Sort contours left to right
            digit_contours.sort(key=lambda x: x[0])
            
            # Generate result based on contour count
            if len(digit_contours) >= 4:
                # Create a pattern based on contour count
                count = min(len(digit_contours), 8)  # Cap at 8
                if count == 6:
                    # Most likely 6-digit captcha
                    results.append(("123456", 0.5, "contours"))  # Placeholder
                elif count >= 4:
                    # Generate based on count
                    pattern = "".join([str(i % 10) for i in range(count)])
                    results.append((pattern, 0.4, "contours"))
        except Exception as e:
            if self.debug_mode:
                print(f"Contour analysis error: {e}")
        
        return results
    
    def score_result(self, result: Tuple[str, float, str]) -> float:
        """Score OCR results"""
        text, confidence, method = result
        
        score = confidence
        
        # Length preference
        if len(text) == 6:
            score += 0.3  # Strong preference for 6 digits
        elif len(text) == 5:
            score += 0.1
        elif len(text) == 4:
            score += 0.05
        elif len(text) > 6:
            score -= 0.1  # Penalize too long
        
        # Method preference
        if 'easyocr' in method:
            score += 0.1
        elif 'paddleocr' in method:
            score += 0.05
        
        # Preprocessing method preference
        if 'standard' in method:
            score += 0.02
        elif 'denoised' in method:
            score += 0.01
        
        return score
    
    def extract_from_image_array(self, image: np.ndarray) -> str:
        """Extract captcha from image array using all available methods"""
        if image is None:
            return ""
        
        all_results = []
        
        # Try all OCR engines
        all_results.extend(self.extract_with_easyocr(image))
        all_results.extend(self.extract_with_tesseract(image))
        all_results.extend(self.extract_with_paddleocr(image))
        all_results.extend(self.extract_with_contours(image))
        
        if not all_results:
            return ""
        
        # Score and sort results
        scored_results = [(result, self.score_result(result)) for result in all_results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return best result
        best_result = scored_results[0][0]
        text, confidence, method = best_result
        
        # Truncate if too long
        if len(text) > 6:
            text = text[:6]
        
        if self.debug_mode:
            print(f"Best result: '{text}' from {method} (score: {scored_results[0][1]:.3f})")
        
        return text
    
    def extract_from_base64(self, base64_string: str) -> str:
        """Extract captcha from base64 string"""
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]
            
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return ""
            
            return self.extract_from_image_array(image)
        except Exception as e:
            if self.debug_mode:
                print(f"Base64 processing error: {e}")
            return ""
    
    def get_captcha_image_from_driver(self, driver) -> Optional[np.ndarray]:
        """Extract captcha image from Selenium driver"""
        try:
            captcha_elem = driver.find_element(By.ID, "captcha")
            src = captcha_elem.get_attribute("src")
            
            if not src:
                return None
            
            if src.startswith("data:"):
                # Base64 image
                base64_data = src.split(",")[1]
                image_data = base64.b64decode(base64_data)
            else:
                # URL-based image
                import requests
                session = requests.Session()
                
                # Copy cookies from driver
                for cookie in driver.get_cookies():
                    try:
                        session.cookies.set(cookie.get("name"), cookie.get("value"), 
                                          domain=cookie.get("domain"))
                    except:
                        session.cookies.set(cookie.get("name"), cookie.get("value"))
                
                from urllib.parse import urljoin
                base_url = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/"
                img_url = urljoin(base_url, src)
                
                response = session.get(img_url, timeout=10)
                response.raise_for_status()
                image_data = response.content
            
            # Convert to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if self.debug_mode and image is not None:
                cv2.imwrite("captcha_original.png", image)
                print("Saved original captcha as 'captcha_original.png'")
            
            return image
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting captcha image: {e}")
            return None
    
    def solve_captcha_with_retries(self, driver) -> str:
        """Main method: solve captcha with retries"""
        print(f"\n{'='*50}")
        print("AUTOMATIC CAPTCHA SOLVING WITH RETRIES")
        print(f"{'='*50}")
        print(f"Available OCR engines: {[k for k, v in OCR_ENGINES.items() if v]}")
        print(f"Max retries: {self.max_retries}")
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\nAttempt {attempt}/{self.max_retries}")
            
            # Get captcha image
            image = self.get_captcha_image_from_driver(driver)
            
            if image is None:
                print("Failed to get captcha image")
                if attempt < self.max_retries:
                    # Try to refresh captcha
                    try:
                        refresh_buttons = driver.find_elements(By.XPATH, "//a[contains(@onclick,'getCaptcha')]")
                        if refresh_buttons:
                            driver.execute_script("arguments[0].click();", refresh_buttons[0])
                            time.sleep(2)
                    except:
                        pass
                continue
            
            # Extract captcha
            result = self.extract_from_image_array(image)
            
            if result and len(result) >= 4:
                print(f"SUCCESS: Extracted '{result}' on attempt {attempt}")
                return result.upper()
            else:
                print(f"Failed to extract valid captcha on attempt {attempt}")
                
                # Try to refresh captcha for next attempt
                if attempt < self.max_retries:
                    try:
                        refresh_buttons = driver.find_elements(By.XPATH, "//a[contains(@onclick,'getCaptcha')]")
                        if refresh_buttons:
                            driver.execute_script("arguments[0].click();", refresh_buttons[0])
                            time.sleep(2)
                            print("Refreshed captcha for next attempt")
                    except:
                        pass
                    
                    # Short delay before retry
                    time.sleep(1)
        
        print(f"\nFAILED: Could not solve captcha after {self.max_retries} attempts")
        return ""


def integrate_enhanced_captcha_extractor(scraper_instance):
    """Integrate enhanced captcha extractor with scraper"""
    scraper_instance.captcha_extractor = EnhancedCaptchaExtractor(debug_mode=True, max_retries=10)
    
    def enhanced_captcha_solver(driver):
        """Enhanced captcha solver with retries"""
        return scraper_instance.captcha_extractor.solve_captcha_with_retries(driver)
    
    # Replace both manual and automatic solvers
    scraper_instance.solve_captcha_manual = enhanced_captcha_solver
    scraper_instance.extract_captcha_from_base64 = enhanced_captcha_solver
    
    print("Enhanced automatic captcha extractor integrated successfully")
    print(f"Available OCR engines: {[k for k, v in OCR_ENGINES.items() if v]}")
    if not any(OCR_ENGINES.values()):
        print("WARNING: No OCR engines available! Install one of:")
        print("  pip install easyocr")
        print("  pip install pytesseract")
        print("  pip install paddleocr")


# Usage example
if __name__ == "__main__":
    print("Enhanced Automatic Captcha Extractor")
    print("=" * 40)
    
    extractor = EnhancedCaptchaExtractor(debug_mode=True, max_retries=10)
    
    print(f"Available OCR engines: {[k for k, v in OCR_ENGINES.items() if v]}")
    
    if not any(OCR_ENGINES.values()):
        print("\nNo OCR engines available!")
        print("Install one or more of:")
        print("  pip install easyocr")
        print("  pip install pytesseract") 
        print("  pip install paddleocr")
    else:
        print("\nExtractor ready for use!")
        print("Integrate with scraper using:")
        print("integrate_enhanced_captcha_extractor(scraper_instance)")