# Windows Test Script for Rajasthan HC Scraper
# Run this to test if everything is working correctly

import sys
import os
from datetime import datetime
from webdriver_manager.chrome import ChromeDriverManager


service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


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
            print(f"⚠️ Website returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot access website: {e}")
        print("💡 Check your internet connection")
        return False

def test_file_operations():
    """Test if file operations work correctly"""
    print("\n🔍 Testing file operations...")
    
    try:
        test_dir = "test_scraper_output"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test CSV writing
        import pandas as pd
        test_data = [{"test_col1": "test_value1", "test_col2": "test_value2"}]
        df = pd.DataFrame(test_data)
        csv_path = os.path.join(test_dir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        # Test reading back
        df_read = pd.read_csv(csv_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        print("✅ File operations working")
        return True
        
    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False

def run_mini_scraper_test():
    """Run a minimal version of the scraper to test functionality"""
    print("\n🔍 Running mini scraper test...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("🌐 Loading Rajasthan HC website...")
        driver.get("https://hcraj.nic.in/cishcraj-jdp/JudgementFilters/")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Check if key form elements exist
        from_date_fields = driver.find_elements(By.NAME, "fromDate")
        to_date_fields = driver.find_elements(By.NAME, "toDate")
        
        print(f"📅 From date field found: {len(from_date_fields) > 0}")
        print(f"📅 To date field found: {len(to_date_fields) > 0}")
        
        # Look for captcha
        captcha_imgs = driver.find_elements(By.XPATH, "//img[contains(@src, 'captcha') or contains(@src, 'Captcha')]")
        print(f"🖼️ Captcha image found: {len(captcha_imgs) > 0}")
        
        driver.quit()
        
        if len(from_date_fields) > 0 and len(to_date_fields) > 0:
            print("✅ Mini scraper test passed - website structure looks correct")
            return True
        else:
            print("⚠️ Website structure might have changed")
            return False
            
    except Exception as e:
        print(f"❌ Mini scraper test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAJASTHAN HC SCRAPER - WINDOWS COMPATIBILITY TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Package Imports", test_imports),
        ("Chrome Driver", test_chrome_driver),
        ("Website Access", test_website_access),
        ("File Operations", test_file_operations),
        ("Mini Scraper", run_mini_scraper_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your system is ready to run the Rajasthan HC scraper")
        print("\n📖 Next steps:")
        print("1. Run the main scraper: python rajasthan_hc_scraper_windows.py")
        print("2. The scraper will prompt for manual captcha input if needed")
        print("3. Check the 'rajasthan_hc_judgments' folder for results")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("🔧 Please fix the issues above before running the scraper")
        print("\n💡 Common solutions:")
        print("1. Install missing packages: pip install [package-name]")
        print("2. Install Chrome browser from https://www.google.com/chrome/")
        print("3. Check internet connection")
        print("4. Install Tesseract OCR for automatic captcha solving")
    
    print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()