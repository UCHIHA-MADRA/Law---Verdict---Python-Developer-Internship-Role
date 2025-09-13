#!/usr/bin/env python3
"""
PaddleOCR Captcha Extraction Utility
Enhanced captcha extraction specifically for Rajasthan High Court website
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import base64
from io import BytesIO
from PIL import Image
import os

class CaptchaExtractor:
    def __init__(self):
        """Initialize PaddleOCR with optimized settings for captcha"""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False,  # Set to True if you have GPU support
                det=True,       # Enable text detection
                rec=True,       # Enable text recognition
                cls=True        # Enable text direction classification
            )
            print("✅ PaddleOCR initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing PaddleOCR: {e}")
            self.ocr = None
    
    def preprocess_captcha_image(self, image):
        """
        Advanced preprocessing for captcha images
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Get original dimensions
            height, width = gray.shape
            
            # Resize small images (captchas are often small)
            if height < 50 or width < 150:
                scale_factor = max(3, 150 // width)
                new_width = width * scale_factor
                new_height = height * scale_factor
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply different thresholding techniques
            methods = []
            
            # Method 1: Otsu's thresholding
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods.append(("Otsu", otsu))
            
            # Method 2: Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            methods.append(("Adaptive", adaptive))
            
            # Method 3: Simple threshold with different values
            for thresh_val in [127, 100, 150]:
                _, simple = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
                methods.append((f"Simple_{thresh_val}", simple))
            
            # Method 4: Inverse thresholding (sometimes captchas have dark text on light background)
            _, inv_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            methods.append(("Inverse_Otsu", inv_otsu))
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            processed_methods = []
            
            for name, img in methods:
                # Remove noise
                cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
                processed_methods.append((name, cleaned))
            
            return processed_methods
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return [("Original", image)]
    
    def extract_from_image_file(self, image_path):
        """
        Extract verification code from image file
        """
        if not self.ocr:
            print("PaddleOCR not initialized")
            return None
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None
            
            return self.extract_from_image_array(image, image_path)
            
        except Exception as e:
            print(f"Error processing image file: {e}")
            return None
    
    def extract_from_base64(self, base64_string):
        """
        Extract verification code from base64 encoded image
        """
        if not self.ocr:
            print("PaddleOCR not initialized")
            return None
        
        try:
            # Remove data URL prefix if present
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Error: Could not decode base64 image")
                return None
            
            return self.extract_from_image_array(image, "base64_image")
            
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None
    
    def extract_from_image_array(self, image, source_name="image"):
        """
        Extract verification code from numpy image array
        """
        if not self.ocr:
            print("PaddleOCR not initialized")
            return None
        
        try:
            print(f"Processing {source_name}...")
            
            # Preprocess image with multiple methods
            processed_images = self.preprocess_captcha_image(image)
            
            all_results = []
            
            for method_name, processed_img in processed_images:
                try:
                    print(f"Trying OCR with {method_name} preprocessing...")
                    
                    # Save processed image for debugging
                    debug_path = f"debug_{method_name}_{source_name}.png"
                    cv2.imwrite(debug_path, processed_img)
                    
                    # Run OCR
                    result = self.ocr.ocr(processed_img, cls=True)
                    
                    if result and result[0]:
                        for line in result[0]:
                            text = line[1][0]
                            confidence = line[1][1]
                            
                            print(f"  {method_name}: '{text}' (confidence: {confidence:.3f})")
                            
                            # Clean text
                            clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                            
                            if clean_text and confidence > 0.5:
                                all_results.append({
                                    'text': clean_text,
                                    'confidence': confidence,
                                    'method': method_name,
                                    'original': text
                                })
                    
                    # Clean up debug file if no good results
                    if os.path.exists(debug_path):
                        if not any(r['confidence'] > 0.7 for r in all_results if r['method'] == method_name):
                            os.remove(debug_path)
                
                except Exception as e:
                    print(f"  Error with {method_name}: {e}")
                    continue
            
            if not all_results:
                print("No text detected with any method")
                return None
            
            # Sort by confidence and filter results
            all_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Look for captcha-like patterns
            captcha_candidates = []
            
            for result in all_results:
                text = result['text']
                confidence = result['confidence']
                
                # Check if it looks like a captcha (4-8 alphanumeric characters)
                if 4 <= len(text) <= 8 and text.isalnum():
                    captcha_candidates.append(result)
            
            # Return best candidate
            if captcha_candidates:
                best = captcha_candidates[0]
                print(f"\n🎯 Best captcha candidate: '{best['text']}' "
                      f"(confidence: {best['confidence']:.3f}, method: {best['method']})")
                return best['text'].upper()
            
            # If no perfect candidates, return highest confidence result
            elif all_results:
                best = all_results[0]
                if best['confidence'] > 0.7:
                    clean_result = re.sub(r'[^A-Za-z0-9]', '', best['text']).upper()
                    print(f"\n🤔 Best guess: '{clean_result}' "
                          f"(confidence: {best['confidence']:.3f}, method: {best['method']})")
                    return clean_result
            
            print("No reliable captcha text found")
            return None
            
        except Exception as e:
            print(f"Error extracting from image array: {e}")
            return None
    
    def extract_from_cropped_region(self, image_path, x, y, w, h):
        """
        Extract verification code from a specific region of the image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None
            
            # Crop the region
            cropped = image[y:y+h, x:x+w]
            
            if cropped.size == 0:
                print("Error: Cropped region is empty")
                return None
            
            # Save cropped image for reference
            cv2.imwrite("cropped_captcha.png", cropped)
            print("Saved cropped region as 'cropped_captcha.png'")
            
            return self.extract_from_image_array(cropped, "cropped_region")
            
        except Exception as e:
            print(f"Error cropping image: {e}")
            return None
    
    def batch_test_with_different_settings(self, image_path):
        """
        Test different OCR settings to find the best one for this type of captcha
        """
        if not self.ocr:
            print("PaddleOCR not initialized")
            return None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None
            
            print(f"Testing different OCR configurations on {image_path}...")
            
            # Test with different PaddleOCR configurations
            configs = [
                {"use_angle_cls": True, "det": True},
                {"use_angle_cls": False, "det": True},
                {"use_angle_cls": True, "det": False},
                {"use_angle_cls": False, "det": False},
            ]
            
            results = []
            
            for i, config in enumerate(configs):
                try:
                    print(f"\nTesting config {i+1}: {config}")
                    
                    # Create temporary OCR instance with different config
                    temp_ocr = PaddleOCR(
                        lang='en',
                        show_log=False,
                        **config
                    )
                    
                    # Process image with multiple preprocessing methods
                    processed_images = self.preprocess_captcha_image(image)
                    
                    for method_name, processed_img in processed_images[:3]:  # Test top 3 methods
                        try:
                            result = temp_ocr.ocr(processed_img, cls=True)
                            
                            if result and result[0]:
                                for line in result[0]:
                                    text = line[1][0]
                                    confidence = line[1][1]
                                    clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                                    
                                    if clean_text and confidence > 0.5:
                                        results.append({
                                            'config': f"Config_{i+1}",
                                            'method': method_name,
                                            'text': clean_text,
                                            'confidence': confidence,
                                            'original': text
                                        })
                                        print(f"  {method_name}: '{clean_text}' (conf: {confidence:.3f})")
                        
                        except Exception as e:
                            print(f"    Error with {method_name}: {e}")
                
                except Exception as e:
                    print(f"  Error with config {i+1}: {e}")
            
            # Summary of results
            if results:
                print(f"\n📊 Summary of {len(results)} results:")
                results.sort(key=lambda x: x['confidence'], reverse=True)
                
                for i, r in enumerate(results[:10]):  # Show top 10
                    print(f"  {i+1}. '{r['text']}' - {r['confidence']:.3f} "
                          f"({r['config']}, {r['method']})")
                
                return results[0]['text'].upper()
            else:
                print("No results found with any configuration")
                return None
                
        except Exception as e:
            print(f"Error in batch testing: {e}")
            return None


def main():
    """Example usage and testing"""
    print("PaddleOCR Captcha Extraction Utility")
    print("=" * 50)
    
    extractor = CaptchaExtractor()
    
    if not extractor.ocr:
        print("PaddleOCR not available. Please install it with:")
        print("pip install paddleocr")
        print("pip install opencv-python")
        return
    
    # Example usage with different input types
    
    # Test with image file
    image_path = "captcha_example.png"
    if os.path.exists(image_path):
        print(f"\n🔍 Testing with image file: {image_path}")
        result = extractor.extract_from_image_file(image_path)
        if result:
            print(f"Extracted: {result}")
        
        # Test batch configurations
        print(f"\n🧪 Testing different configurations...")
        batch_result = extractor.batch_test_with_different_settings(image_path)
        if batch_result:
            print(f"Best batch result: {batch_result}")
    
    # Test with base64 (example from Rajasthan HC website)
    base64_example = "iVBORw0KGgoAAAANSUhEUgAAANIAAAAoCAMAAAC8TlQPAAAAM1BMVEUAAP/////mLIX///8/P/+fn//f3/+/v/9fX/8fH/9/f//ylcLpRpTve7L1r9H4yuD75O+YOONGAAAACXBIWXMAAA7EAAAOxAGVKw4bAAADfElEQVRoge1Z23LcIAzdAUkgmrT9/68twmCDEb6tk/ohZyaTzWKkozs4r5cC0324CDP4/INvx52hvF3F1+XGF2bdEdE/SX8G93vrdolDgUDwbSSu6ThLw1vrjjzHRx4yOoU9Tmb4xx4YApJvvwu2NcnUMpdPnpyqDtC5sGvtIXdoAKRIr6ZcMfDo4moENrww7rC05xnzctZqVF0SaX9v7GVhRftm95RTBiWoacTTEvpSN6ZQAogm7UmX7dDabeKfkWwA7yadulsKKzoZKZEGMQAQfRK16Cb5TiiQaIpLu+okPXux8VvIv3GwMQqX7rPxxAacRTP91pyu8ZbGwGlJbXm116NQ2xfTzBMHItJK0hsO+G0d5pmY17N++rYu/tlz1AZASSC25HtOiy8kFjpNl3XA0OgxgHBSuWUSQ0XLF0tcV35rq4INkrlrlQtNklWtmN4xaQaNaukVpOVVc7WYF1QntDIh/qyrgUt9pYaohsmUxMNRGAeoezXp3YelWVsSo36t11Ye7AdlzDujkVrKllwfxKKXgAHt+IjSa64ZpFatzgCQicSpyfW6+wC0YoNEA/pi8jlMJljfZ29WPE1D6cbXQIhOnQFQegFT723cyTxKxii++JP8xCGK1EtYULIDT5101iO3dxg4XJZbhxmxV82asndyglQ6184y4p8JPBwFkp2ycG0yFZQJqHslF3VzwKTFCcomnNIrSDmuaWE6lHCSoboF7d+PhtUlsNICKix9asbnpjrJOwgunxDXVkMeDAOTllOHfqzpoAcix3iQu1qXp40LRsw7LDW+oVsavdppi6V9iEeilPW1P9odTeHkpTAala988CT0PGpq5Sn1+4XLhkkaTcGH97P0Po3Yf86rSpOnrk3OCjheWHjM2pMwBSX2+cjpihjrz15PS/2xmkXVKir+mJv8hla9qQUJoRtejX2Jvzt2v2j0c7q9QDw/9GEw1WonOUnB4XhflDSRqI+/6tUjI1obb2lBG/FDRTPKlNDJlTFeWdR6ZEdhRAhstJ0+rAuxvj3mm+/6DcFBeEfx4jpY5OCI8POS4AanX9igG7M6oubWF1WXhJ3b9Abf04re983D3gX/Jzo3qL2J+TPeNh/c84zkeYvF+c3PMLrCQwidemt+evcZ3PwP3of4V8Nb4+w+u3YlHfxHUP/Y8Ky0r+zmsJl75G1K+QcDzhJ+ug4NrgAAAABJRU5ErkJggg=="
    
    print(f"\n🔍 Testing with base64 data...")
    result = extractor.extract_from_base64(base64_example)
    if result:
        print(f"Extracted from base64: {result}")
    
    print(f"\n✅ Testing completed!")


if __name__ == "__main__":
    main()