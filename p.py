# Windows Test Script for Rajasthan HC Scraper
# Run this to test if everything is working correctly

import sys
import os
from datetime import datetime

def test_imports():
    """Test if all required packages are available"""
    print("🔍 Testing imports...")
    
    try:
        import selenium
        print(f"✅ Selenium: {selenium.__version__}")
    except ImportError as e:
        print(f"❌ Selenium not found: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas not found: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV not found: {e}")
        return False
    
    try:
        import requests
        print(f"✅ Requests: {requests.__version__}")
    except ImportError as e:
        print(f"❌ Requests not found: {e}")
        return False
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ WebDriver Manager: Available")
    except ImportError as e:
        print(f"❌ WebDriver Manager not found: {e}")
        return False
    
    try:
        import pytesseract
        print("✅ Pytesseract: Available")
    except ImportError:
        print("⚠️ Pytesseract not found (captcha will be manual)")
    
    return True

def test_chrome_driver():
    """Test if Chrome driver can be set up"""
    print("\n🔍 Testing Chrome driver setup...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test basic functionality
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print(f"✅ Chrome driver working! Test page title: {title}")
        return True
        
    except Exception as e:
        print(f"❌ Chrome driver test failed: {e}")
        print("💡 Make sure Google Chrome browser is installed")
        return False

def test_website_access():
    """Test if we can access the target website"""
    print("\n🔍 Testing website access...")
    
    try:
        import requests
        
        url = "https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Website accessible")
            return True
        else:
            print(f"⚠️ Website returned status code: {response.status_code