#!/usr/bin/env python3
"""
Test script for Rajasthan High Court Judgment Scraper

This script tests the functionality of the scraper, including:
1. Incremental daily download
2. Date range handling
3. CSV output
4. Captcha solving
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import shutil

# Import the scraper
from RajasthanHighCourtJudgmentScraper import FixedRajasthanHCScraper, SCICaptchaSolver

class TestRajasthanHCScraper(unittest.TestCase):
    """
    Test cases for Rajasthan High Court Judgment Scraper
    """
    
    def setUp(self):
        """Set up test environment"""
        # Create a test directory
        self.test_dir = Path("test_rajasthan_hc_judgments")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test PDF directory
        self.test_pdf_dir = self.test_dir / "pdfs"
        self.test_pdf_dir.mkdir(exist_ok=True)
        
        # Initialize the scraper with test directory
        self.scraper = FixedRajasthanHCScraper(download_dir=str(self.test_dir))
        
        # Create a sample judgment for testing
        self.sample_judgment = {
            'judgment_id': 'test_id_123',
            'case_number': 'TEST/123/2023',
            'petitioner': 'Test Petitioner',
            'respondent': 'Test Respondent',
            'judgment_date': '01/01/2023',
            'judge_name': 'Test Judge',
            'pdf_url': 'https://example.com/test.pdf',
            'pdf_path': str(self.test_pdf_dir / 'test.pdf'),
            'pdf_filename': 'test.pdf',
            'download_status': 'success',
            'scraped_date': datetime.now().isoformat()
        }
        
        # Create a dummy PDF file
        with open(self.sample_judgment['pdf_path'], 'w') as f:
            f.write('Dummy PDF content')
    
    def tearDown(self):
        """Clean up after tests"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_save_to_csv(self):
        """Test saving judgments to CSV"""
        # Save a sample judgment to CSV
        self.scraper.save_to_csv([self.sample_judgment])
        
        # Check if CSV file exists
        self.assertTrue(self.scraper.csv_file.exists())
        
        # Read the CSV file
        df = pd.read_csv(self.scraper.csv_file)
        
        # Check if the judgment is in the CSV
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['judgment_id'], self.sample_judgment['judgment_id'])
        
        # Check if file_size_kb was added
        self.assertIn('file_size_kb', df.columns)
        
        # Add another judgment and check for duplicates
        judgment2 = self.sample_judgment.copy()
        judgment2['judgment_id'] = 'test_id_456'
        judgment2['case_number'] = 'TEST/456/2023'
        
        self.scraper.save_to_csv([judgment2])
        
        # Read the CSV file again
        df = pd.read_csv(self.scraper.csv_file)
        
        # Check if both judgments are in the CSV
        self.assertEqual(len(df), 2)
        
        # Test updating an existing judgment
        updated_judgment = self.sample_judgment.copy()
        updated_judgment['download_status'] = 'updated'
        
        self.scraper.save_to_csv([updated_judgment])
        
        # Read the CSV file again
        df = pd.read_csv(self.scraper.csv_file)
        
        # Check if the judgment was updated and not duplicated
        self.assertEqual(len(df), 2)
        updated_row = df[df['judgment_id'] == self.sample_judgment['judgment_id']]
        self.assertEqual(updated_row.iloc[0]['download_status'], 'updated')
    
    def test_date_range_handling(self):
        """Test date range handling"""
        # Mock the scrape_judgments method
        with patch.object(self.scraper, 'scrape_judgments', return_value=[self.sample_judgment]) as mock_scrape:
            # Run incremental scrape with default 10 days
            self.scraper.run_incremental_scrape()
            
            # Check if scrape_judgments was called with correct date range
            today = datetime.now()
            from_date = (today - timedelta(days=10)).strftime("%d/%m/%Y")
            to_date = today.strftime("%d/%m/%Y")
            
            mock_scrape.assert_called_once()
            args, _ = mock_scrape.call_args
            self.assertEqual(args[0], from_date)
            self.assertEqual(args[1], to_date)
    
    def test_incremental_download(self):
        """Test incremental download functionality"""
        # Mock the scrape_judgments method
        with patch.object(self.scraper, 'scrape_judgments', return_value=[self.sample_judgment]) as mock_scrape:
            # First run
            self.scraper.run_incremental_scrape()
            
            # Check if state was updated
            self.assertIn('last_run_date', self.scraper.downloaded_judgments)
            self.assertIn(self.sample_judgment['judgment_id'], self.scraper.downloaded_judgments['downloaded_ids'])
            
            # Second run with same judgment
            mock_scrape.reset_mock()
            self.scraper.run_incremental_scrape()
            
            # Check if scrape_judgments was called again (it should be)
            mock_scrape.assert_called_once()
            
            # The scrape_judgments method should skip already downloaded judgments
            # This is tested in the actual implementation
    
    def test_captcha_solver(self):
        """Test captcha solver functionality"""
        # Initialize the SCI captcha solver
        solver = SCICaptchaSolver()
        
        # Check if the solver was initialized
        self.assertIsNotNone(solver)
        
        # Test preprocessing methods
        # This is a basic test to ensure the methods don't crash
        # Create a dummy captcha image
        import numpy as np
        import cv2
        
        dummy_image = np.zeros((50, 150, 3), dtype=np.uint8)
        dummy_image_path = str(self.test_dir / 'dummy_captcha.png')
        cv2.imwrite(dummy_image_path, dummy_image)
        
        # Test preprocessing
        processed_images = solver.preprocess_sci_captcha(dummy_image_path)
        
        # Check if preprocessing returned images
        self.assertIsInstance(processed_images, list)

if __name__ == '__main__':
    unittest.main()