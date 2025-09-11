#!/usr/bin/env python3
"""
Advanced OCR Captcha Solver for Rajasthan High Court
Specialized for detecting numbers and text from captcha images
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os
import re
from typing import List, Tuple, Optional

class AdvancedCaptchaOCR:
    def __init__(self):
        self.debug = True
        # Set Tesseract path if needed (Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def preprocess_image(self, image_path: str) -> List[Tuple[str, np.ndarray]]:
        """Advanced image preprocessing specifically for captcha OCR"""
        try:
            # Load image
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return []
            
            # Load with PIL first
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
            dilated = cv2.dilate(eroded, kernel, iterations=2)
            processed_images.append(("Erode_Dilate", dilated))
            
            if self.debug:
                print(f"Generated {len(processed_images)} processed images")
                
            return processed_images
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return []
    
    def extract_text_with_multiple_configs(self, image: np.ndarray, method_name: str) -> List[Tuple[str, float]]:
        """Extract text using multiple OCR configurations"""
        ocr_configs = [
            # Basic configurations
            {
                'config': '--oem 3 --psm 7',
                'description': 'Single text line'
            },
            {
                'config': '--oem 3 --psm 8', 
                'description': 'Single word'
            },
            {
                'config': '--oem 3 --psm 6',
                'description': 'Single uniform block'
            },
            {
                'config': '--oem 3 --psm 13',
                'description': 'Raw line'
            },
            
            # With character whitelist (numbers and letters)
            {
                'config': '--oem 3 --psm 7 -c tesseract_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                'description': 'Numbers and letters only'
            },
            {
                'config': '--oem 3 --psm 8 -c tesseract_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                'description': 'Single word - numbers and letters'
            },
            
            # Numbers only
            {
                'config': '--oem 3 --psm 7 -c tesseract_char_whitelist=0123456789',
                'description': 'Numbers only'
            },
            {
                'config': '--oem 3 --psm 8 -c tesseract_char_whitelist=0123456789',
                'description': 'Single number'
            },
            
            # Different OEM modes
            {
                'config': '--oem 1 --psm 7',
                'description': 'Legacy engine'
            },
            {
                'config': '--oem 2 --psm 7',
                'description': 'LSTM only'
            },
            
            # With additional parameters
            {
                'config': '--oem 3 --psm 7 -c preserve_interword_spaces=0',
                'description': 'No spaces'
            },
            {
                'config': '--oem 3 --psm 8 -c load_system_dawg=false -c load_freq_dawg=false',
                'description': 'No dictionary'
            }
        ]
        
        results = []
        
        for config_dict in ocr_configs:
            try:
                config = config_dict['config']
                description = config_dict['description']
                
                # Perform OCR
                raw_text = pytesseract.image_to_string(image, config=config).strip()
                
                # Clean the text
                cleaned_text = self.clean_ocr_result(raw_text)
                
                if cleaned_text and len(cleaned_text) >= 3:
                    confidence = self.calculate_confidence_score(cleaned_text, raw_text)
                    results.append((cleaned_text, confidence))
                    
                    if self.debug:
                        print(f"  {method_name} + {description}: '{cleaned_text}' (confidence: {confidence:.1f})")
                
            except Exception as e:
                if self.debug:
                    print(f"  OCR error with config {config_dict['description']}: {e}")
                continue
        
        return results
    
    def clean_ocr_result(self, raw_text: str) -> str:
        """Clean and normalize OCR result"""
        if not raw_text:
            return ""
        
        # Remove common OCR mistakes and noise
        cleaned = raw_text.strip()
        
        # Remove line breaks and extra spaces
        cleaned = re.sub(r'\s+', '', cleaned)
        
        # Remove common OCR noise characters
        noise_chars = ['|', '\\', '/', '_', '-', '~', '`', "'", '"', '.', ',', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}', '<', '>', '?', ';', ':']
        for char in noise_chars:
            cleaned = cleaned.replace(char, '')
        
        # Convert to uppercase for consistency
        cleaned = cleaned.upper()
        
        # Fix common character recognition errors
        replacements = {
            'O': '0',  # Letter O to number 0
            'I': '1',  # Letter I to number 1
            'S': '5',  # Letter S to number 5 (sometimes)
            'Z': '2',  # Letter Z to number 2 (sometimes)
            'G': '6',  # Letter G to number 6 (sometimes)
            'B': '8',  # Letter B to number 8 (sometimes)
        }
        
        # Apply replacements if the result looks like it should be mostly numbers
        if len(cleaned) >= 4 and sum(c.isdigit() for c in cleaned) / len(cleaned) > 0.5:
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
        
        # Only return if it's a reasonable length and contains alphanumeric characters
        if 3 <= len(cleaned) <= 10 and cleaned.isalnum():
            return cleaned
        
        return ""
    
    def calculate_confidence_score(self, cleaned_text: str, raw_text: str) -> float:
        """Calculate confidence score for OCR result"""
        score = 0.0
        
        # Length score (captchas are usually 4-6 characters)
        if 4 <= len(cleaned_text) <= 6:
            score += 30
        elif len(cleaned_text) == 3 or len(cleaned_text) == 7:
            score += 15
        elif len(cleaned_text) == 8:
            score += 10
        
        # Character composition
        digit_ratio = sum(c.isdigit() for c in cleaned_text) / len(cleaned_text)
        letter_ratio = sum(c.isalpha() for c in cleaned_text) / len(cleaned_text)
        
        # Mixed numbers and letters is common in captchas
        if 0.2 <= digit_ratio <= 0.8 and 0.2 <= letter_ratio <= 0.8:
            score += 25
        elif digit_ratio == 1.0 or letter_ratio == 1.0:
            score += 15  # All digits or all letters
        
        # Penalize very short or very long results
        if len(cleaned_text) < 3 or len(cleaned_text) > 8:
            score -= 20
        
        # Check if cleaning process removed a lot of characters (indicates noise)
        if len(raw_text) > 0:
            cleaning_ratio = len(cleaned_text) / len(raw_text.strip())
            if cleaning_ratio > 0.7:
                score += 10  # Good cleaning ratio
            elif cleaning_ratio < 0.3:
                score -= 15  # Too much was removed
        
        # Bonus for no repeated characters (less common in noise)
        if len(set(cleaned_text)) == len(cleaned_text):
            score += 10
        
        # Penalty for too many repeated characters
        if len(set(cleaned_text)) < len(cleaned_text) / 2:
            score -= 15
        
        return max(0.0, score)
    
    def solve_captcha(self, image_path: str) -> str:
        """Main function to solve captcha from image path"""
        print(f"Attempting to solve captcha: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Captcha image not found: {image_path}")
            return ""
        
        # Preprocess the image
        processed_images = self.preprocess_image(image_path)
        
        if not processed_images:
            print("Failed to preprocess image")
            return ""
        
        all_results = []
        
        # Try OCR on each processed image
        for method_name, processed_img in processed_images:
            if self.debug:
                print(f"\nTrying OCR on {method_name}...")
                
                # Save debug image
                debug_filename = f"debug_{method_name.lower()}.png"
                cv2.imwrite(debug_filename, processed_img)
            
            # Extract text with multiple configurations
            method_results = self.extract_text_with_multiple_configs(processed_img, method_name)
            all_results.extend(method_results)
        
        if not all_results:
            print("No valid OCR results found")
            return ""
        
        # Sort results by confidence score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"\nAll results ({len(all_results)} total):")
            for i, (text, confidence) in enumerate(all_results[:10]):  # Show top 10
                print(f"  {i+1}. '{text}' (confidence: {confidence:.1f})")
        
        # Filter results by minimum confidence
        min_confidence = 20.0
        good_results = [(text, conf) for text, conf in all_results if conf >= min_confidence]
        
        if not good_results:
            print(f"No results above confidence threshold {min_confidence}")
            if all_results:
                best_result = all_results[0][0]
                print(f"Returning best available result: '{best_result}'")
                return best_result
            return ""
        
        # Look for consensus among top results
        top_results = [text for text, _ in good_results[:5]]
        
        # Find most common result
        from collections import Counter
        result_counts = Counter(top_results)
        most_common = result_counts.most_common(1)
        
        if most_common and most_common[0][1] > 1:
            # Multiple methods agree
            consensus_result = most_common[0][0]
            print(f"Consensus result: '{consensus_result}' (appears {most_common[0][1]} times)")
            return consensus_result
        else:
            # Use highest confidence result
            best_result = good_results[0][0]
            best_confidence = good_results[0][1]
            print(f"Best result: '{best_result}' (confidence: {best_confidence:.1f})")
            return best_result

def test_captcha_solver():
    """Test the captcha solver on available images"""
    solver = AdvancedCaptchaOCR()
    
    test_images = ["manual_captcha.png", "captcha_temp.png", "temp_captcha.png"]
    
    for image_file in test_images:
        if os.path.exists(image_file):
            print(f"\n{'='*50}")
            print(f"Testing on: {image_file}")
            print('='*50)
            
            result = solver.solve_captcha(image_file)
            
            if result:
                print(f"\nFINAL RESULT: '{result}'")
            else:
                print("\nNo solution found")
            
            return result  # Return first successful result
    
    print("No captcha images found to test")
    return ""

if __name__ == "__main__":
    print("ADVANCED CAPTCHA OCR SOLVER")
    print("="*50)
    test_captcha_solver()