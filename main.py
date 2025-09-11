#!/usr/bin/env python3
"""
Rajasthan High Court Judgment Scraper - Main Script
This script implements the requirements for downloading judgments from Rajasthan High Court
with incremental daily updates.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from RajasthanHighCourtJudgmentScraper import FixedRajasthanHCScraper, SCICaptchaSolver

def main():
    """
    Main function to run the Rajasthan High Court Judgment Scraper
    
    Features:
    1. Downloads judgments from the last 10 days by default
    2. Implements incremental daily download functionality
    3. Saves all relevant information to a CSV file
    4. Downloads PDF judgments
    5. Handles captcha solving
    """
    parser = argparse.ArgumentParser(description="Rajasthan High Court Judgment Scraper")
    parser.add_argument(
        "--days", 
        type=int, 
        default=10, 
        help="Number of days to look back (default: 10)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="rajasthan_hc_judgments", 
        help="Output directory for judgments and CSV (default: rajasthan_hc_judgments)"
    )
    parser.add_argument(
        "--test-sci-captcha", 
        action="store_true", 
        help="Test the Supreme Court of India captcha solver"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAJASTHAN HIGH COURT JUDGMENT SCRAPER")
    print("=" * 80)
    print(f"Looking back {args.days} days from today")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Initialize the scraper
    scraper = FixedRajasthanHCScraper(download_dir=args.output_dir)
    
    # Run the incremental scrape
    try:
        judgments = scraper.run_incremental_scrape(days_back=args.days)
        
        # Display summary
        if scraper.csv_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(scraper.csv_file)
                print("\nSummary:")
                print(f"Total judgments in database: {len(df)}")
                
                if 'download_status' in df.columns:
                    status_counts = df['download_status'].value_counts()
                    print("\nDownload status:")
                    for status, count in status_counts.items():
                        print(f"  {status}: {count}")
            except Exception as e:
                print(f"Error generating summary: {e}")
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during scraping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test SCI captcha solver if requested
    if args.test_sci_captcha:
        print("\n" + "=" * 80)
        print("TESTING SUPREME COURT OF INDIA CAPTCHA SOLVER")
        print("=" * 80)
        
        sci_solver = SCICaptchaSolver()
        
        # Look for captcha images to test
        test_images = ["manual_captcha.png", "captcha_temp.png"]
        for img in test_images:
            if os.path.exists(img):
                print(f"Testing with image: {img}")
                result = sci_solver.solve_sci_captcha(img)
                print(f"Result: {result}")
                break
        else:
            print("No captcha images found for testing")
    
    print("\nScript completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())