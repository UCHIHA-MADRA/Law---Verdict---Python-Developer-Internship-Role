#!/usr/bin/env python3
"""
Numerical Captcha Extractor for Rajasthan High Court
Optimized for 6-digit numerical captcha recognition
"""

import cv2
import numpy as np
import re
import base64
import logging
from typing import Optional, List, Tuple, Dict, Any, Union
from urllib.parse import urljoin

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False

class NumericalCaptchaExtractor:
    """Specialized captcha extractor for 6-digit numerical codes"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the numerical captcha extractor
        
        Args:
            debug_mode: If True, saves debug images and prints verbose output
        """
        self.ocr: Optional[Any] = None
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en', 
                    show_log=False,
                    use_gpu=False,
                    det=True,
                    rec=True,
                    cls=True
                ) # pyright: ignore[reportOptionalCall]
                if self.debug_mode:
                    print("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.ocr = None
        else:
            if self.debug_mode:
                print("PaddleOCR not available - install with: pip install paddleocr")
    
    def is_available(self) -> bool:
        """Check if OCR is available for use"""
        return self.ocr is not None
    
    def preprocess_image(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Apply various preprocessing techniques optimized for numerical captchas
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (method_name, processed_image) tuples
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize for better OCR accuracy
        height, width = gray.shape
        if width < 200 or height < 50:
            scale_factor = max(2, 200 // width)
            new_width = width * scale_factor
            new_height = height * scale_factor
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            if self.debug_mode:
                print(f"Resized from {width}x{height} to {new_width}x{new_height}")
        
        methods = []
        
        # Method 1: Gaussian blur + Otsu thresholding
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("gaussian_otsu", otsu))
        
        # Method 2: Bilateral filter + adaptive threshold
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        methods.append(("bilateral_adaptive", adaptive))
        
        # Method 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        methods.append(("morphological", morph))
        
        # Method 4: Histogram equalization
        equalized = cv2.equalizeHist(gray)
        _, eq_thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("equalized", eq_thresh))
        
        # Check if images need inversion (dark background)
        final_methods = []
        for name, img in methods:
            # Invert if background is dark
            if img.mean() < 127:
                inverted = cv2.bitwise_not(img)
                final_methods.append((f"{name}_inverted", inverted))
            final_methods.append((name, img))
        
        return final_methods
    
    def correct_ocr_errors(self, text: str) -> str:
        """
        Correct common OCR errors for numerical captchas
        
        Args:
            text: Raw OCR output
            
        Returns:
            Corrected text with common substitutions
        """
        # Common OCR misreads for digits
        corrections = {
            'O': '0', 'o': '0', 'Q': '0', 'D': '0',
            'I': '1', 'l': '1', '|': '1', 'i': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'G': '6', 'g': '6',
            'T': '7', 't': '7',
            'B': '8', 'b': '8',
            'g': '9'
        }
        
        corrected = text
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        
        return corrected
    
    def validate_numerical_result(self, text: str) -> Tuple[bool, str]:
        """
        Validate and clean numerical captcha result
        
        Args:
            text: OCR output text
            
        Returns:
            Tuple of (is_valid, cleaned_text)
        """
        if not text:
            return False, ""
        
        # Apply OCR error corrections
        corrected = self.correct_ocr_errors(text)
        
        # Extract only digits
        digits_only = re.sub(r'[^0-9]', '', corrected)
        
        # Check for exactly 6 digits (ideal case)
        if len(digits_only) == 6:
            return True, digits_only
        
        # If we have more than 6 digits, try to find the best 6-digit sequence
        if len(digits_only) > 6:
            # Look for consecutive 6-digit sequences
            for i in range(len(digits_only) - 5):
                candidate = digits_only[i:i+6]
                if len(candidate) == 6:
                    return True, candidate
        
        # Accept 4-5 digits as partial matches (might be useful)
        if 4 <= len(digits_only) <= 5:
            return True, digits_only
        
        return False, ""
    
    def extract_from_image_array(self, image: np.ndarray) -> str:
        """
        Extract numerical captcha from image array
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Extracted captcha text or empty string if failed
        """
        if not self.ocr:
            return ""
        
        try:
            # Preprocess image with multiple methods
            processed_images = self.preprocess_image(image)
            candidates = []
            
            for method_name, processed_img in processed_images:
                try:
                    # Save debug image if in debug mode
                    if self.debug_mode:
                        cv2.imwrite(f"debug_{method_name}.png", processed_img)
                    
                    # Run OCR with compatibility handling
                    try:
                        # Try with cls parameter (newer versions)
                        results = self.ocr.ocr(processed_img, cls=True)
                    except TypeError:
                        # Fallback for older versions without cls parameter
                        results = self.ocr.ocr(processed_img)
                    
                    if results and results[0]:
                        for detection in results[0]:
                            if len(detection) >= 2:
                                text_info = detection[1]
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    text = str(text_info[0])
                                    confidence = float(text_info[1])
                                    
                                    # Validate result
                                    is_valid, cleaned = self.validate_numerical_result(text)
                                    
                                    if is_valid and confidence > 0.3:
                                        candidates.append({
                                            'text': cleaned,
                                            'confidence': confidence,
                                            'method': method_name,
                                            'original': text
                                        })
                                        
                                        if self.debug_mode:
                                            print(f"{method_name}: '{text}' -> '{cleaned}' (conf: {confidence:.3f})")
                
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error with method {method_name}: {e}")
                    continue
            
            if not candidates:
                return ""
            
            # Score and sort candidates
            def calculate_score(candidate: Dict) -> float:
                score = candidate['confidence']
                # Bonus for exactly 6 digits
                if len(candidate['text']) == 6:
                    score += 0.3
                elif len(candidate['text']) == 5:
                    score += 0.1
                return score
            
            candidates.sort(key=calculate_score, reverse=True)
            best = candidates[0]
            
            # Return best result if confidence is reasonable
            if len(best['text']) == 6 and best['confidence'] > 0.5:
                if self.debug_mode:
                    print(f"Best result: {best['text']} (confidence: {best['confidence']:.3f})")
                return best['text']
            elif best['confidence'] > 0.7:
                if self.debug_mode:
                    print(f"Partial result: {best['text']} (confidence: {best['confidence']:.3f})")
                return best['text']
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting from image: {e}")
            return ""
    
    def extract_from_base64(self, base64_string: str) -> str:
        """
        Extract captcha from base64 encoded image
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            Extracted captcha text or empty string
        """
        if not self.ocr:
            return ""
        
        try:
            # Remove data URL prefix if present
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]
            
            # Decode base64 to image
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                if self.debug_mode:
                    print("Failed to decode base64 image")
                return ""
            
            return self.extract_from_image_array(image)
            
        except Exception as e:
            self.logger.error(f"Error processing base64 image: {e}")
            return ""
    
    def extract_from_file(self, image_path: str) -> str:
        """
        Extract captcha from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted captcha text or empty string
        """
        if not self.ocr:
            return ""
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                if self.debug_mode:
                    print(f"Could not load image: {image_path}")
                return ""
            
            return self.extract_from_image_array(image)
            
        except Exception as e:
            self.logger.error(f"Error processing image file: {e}")
            return ""
    
    def extract_from_selenium_element(self, driver, element_id: str = "captcha") -> str:
        """
        Extract captcha directly from Selenium WebDriver element
        
        Args:
            driver: Selenium WebDriver instance
            element_id: ID of the captcha image element
            
        Returns:
            Extracted captcha text or empty string
        """
        if not self.ocr:
            return ""
        
        try:
            from selenium.webdriver.common.by import By
            import requests
            
            captcha_element = driver.find_element(By.ID, element_id)
            src = captcha_element.get_attribute("src")
            
            if not src:
                return ""
            
            if src.startswith("data:"):
                return self.extract_from_base64(src)
            else:
                # Fetch image with session cookies
                session = requests.Session()
                for cookie in driver.get_cookies():
                    session.cookies.set(cookie['name'], cookie['value'])
                
                response = session.get(src, timeout=10)
                response.raise_for_status()
                
                # Convert to base64 and process
                b64_data = base64.b64encode(response.content).decode()
                return self.extract_from_base64(b64_data)
                
        except Exception as e:
            self.logger.error(f"Error extracting from Selenium element: {e}")
            return ""


def integrate_with_rajasthan_scraper(scraper_instance):
    """
    Helper function to integrate numerical captcha extractor with existing scraper
    
    Args:
        scraper_instance: Instance of RajasthanHCScraper
    """
    # Add numerical captcha extractor to scraper instance
    scraper_instance.numerical_extractor = NumericalCaptchaExtractor(debug_mode=True)
    
    # Replace the extract_captcha_from_base64 method
    def enhanced_extract_captcha_from_base64(driver):
        """Enhanced captcha extraction for 6-digit numerical captchas"""
        if not hasattr(scraper_instance, 'numerical_extractor'):
            scraper_instance.numerical_extractor = NumericalCaptchaExtractor()
        
        if not scraper_instance.numerical_extractor.is_available():
            # Fallback to manual input
            return scraper_instance.solve_captcha_manual(driver)
        
        try:
            from selenium.webdriver.common.by import By
            
            captcha_elem = driver.find_element(By.ID, "captcha")
            src = captcha_elem.get_attribute("src")
            
            if not src:
                return ""
            
            if src.startswith("data:"):
                result = scraper_instance.numerical_extractor.extract_from_base64(src)
            else:
                # Handle URL-based captcha images
                import requests
                
                sess = requests.Session()
                for c in driver.get_cookies():
                    try:
                        sess.cookies.set(c.get("name"), c.get("value"), domain=c.get("domain"))
                    except:
                        sess.cookies.set(c.get("name"), c.get("value"))
                
                img_url = urljoin(scraper_instance.base_url, src)
                resp = sess.get(img_url, timeout=10)
                resp.raise_for_status()
                
                b64_data = base64.b64encode(resp.content).decode()
                result = scraper_instance.numerical_extractor.extract_from_base64(b64_data)
            
            if result and len(result) >= 4:  # Accept 4+ digit results
                print(f"Auto-extracted captcha: {result}")
                return result.upper()
            else:
                print("Auto-extraction failed, falling back to manual input")
                return scraper_instance.solve_captcha_manual(driver)
                
        except Exception as e:
            try:
                scraper_instance.logger.debug(f"Enhanced captcha extraction error: {e}")
            except:
                print(f"Enhanced captcha extraction error: {e}")
            
            # Fallback to manual input on any error
            return scraper_instance.solve_captcha_manual(driver)
    
    # Replace the method
    scraper_instance.extract_captcha_from_base64 = enhanced_extract_captcha_from_base64
    
    print("Numerical captcha extractor integrated successfully")


def test_extractor():
    """Test the numerical captcha extractor"""
    extractor = NumericalCaptchaExtractor(debug_mode=True)
    
    if not extractor.is_available():
        print("PaddleOCR not available. Install with: pip install paddleocr opencv-python")
        return
    
    print("Testing numerical captcha extractor...")
    
    # Test validation function
    test_cases = [
        "256914",           # Perfect case
        "25691O",          # O instead of 0
        "2S6g14",          # S and g substitutions
        "abc256914xyz",    # Embedded digits
        "25-69-14",        # With separators
        "12345",           # 5 digits
        "1234567",         # 7 digits
    ]
    
    print("Testing validation:")
    for test in test_cases:
        is_valid, cleaned = extractor.validate_numerical_result(test)
        print(f"'{test}' -> Valid: {is_valid}, Cleaned: '{cleaned}'")
    
    # Test with sample base64 (if you have one)
    sample_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANIAAAAoCAMAAAC8TlQP..."
    if len(sample_b64) > 50:
        result = extractor.extract_from_base64(sample_b64)
        print(f"Base64 extraction result: '{result}'")


if __name__ == "__main__":
    test_extractor()