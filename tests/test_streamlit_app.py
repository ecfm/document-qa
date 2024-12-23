import unittest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
from streamlit_app import (
    MAX_REVIEWS, 
    MAX_CHARS_PER_REVIEW,
    validate_review_length,
    validate_reviews_count,
    process_text_input,
    process_reviews
)

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_reviews = [
            "This is a great product!",
            "Not satisfied with the quality.",
            "Would recommend to others."
        ]
        self.long_review = "x" * (MAX_CHARS_PER_REVIEW + 100)
        
    def test_validate_review_length(self):
        """Test review length validation function."""
        # Test normal review
        normal_review = "This is a normal review"
        self.assertEqual(validate_review_length(normal_review), normal_review)
        
        # Test long review
        truncated = validate_review_length(self.long_review)
        self.assertEqual(len(truncated), MAX_CHARS_PER_REVIEW)
        
    def test_validate_reviews_count(self):
        """Test reviews count validation function."""
        # Test normal list
        normal_list = ["Review"] * 5
        self.assertEqual(len(validate_reviews_count(normal_list)), 5)
        
        # Test oversized list
        large_list = ["Review"] * (MAX_REVIEWS + 10)
        self.assertEqual(len(validate_reviews_count(large_list)), MAX_REVIEWS)
        
    def test_process_text_input(self):
        """Test text input processing."""
        # Test single review
        single_review = "This is a test review"
        df, review_column = process_text_input(single_review)
        self.assertEqual(len(df), 1)
        self.assertEqual(df[review_column].iloc[0], single_review)
        
        # Test multiple reviews
        multi_review = "Review 1\nReview 2\nReview 3"
        df, review_column = process_text_input(multi_review)
        self.assertEqual(len(df), 3)
        
    def test_process_reviews(self):
        """Test review processing with mocked API."""
        # Create test dataframe
        input_df = pd.DataFrame({
            "Review": self.sample_reviews
        })
        
        # Mock all the required components
        with patch('streamlit.empty') as mock_empty, \
             patch('streamlit.progress') as mock_progress, \
             patch('requests.post') as mock_post:
            
            # Setup mocks
            mock_empty.return_value = MagicMock()
            mock_progress.return_value = MagicMock()
            
            # Mock the API response
            mock_response = MagicMock()
            mock_response.json.return_value = {"generation": "Mocked API response"}
            mock_post.return_value = mock_response
            
            # Call the function
            output_df = process_reviews(input_df, "Review", "Test Category")
            
            # Debug print
            print("\nOutput DataFrame:")
            print(output_df)
            print("\nColumns:", output_df.columns.tolist())
            print("\nFirst row LLM Response:", repr(output_df['LLM Response'].iloc[0]))
            
            # Verify output
            self.assertEqual(len(output_df), len(self.sample_reviews))
            self.assertIn('LLM Response', output_df.columns)
            self.assertEqual(output_df['LLM Response'].iloc[0], "Mocked API response")
            
            # Verify API calls
            self.assertEqual(mock_post.call_count, len(self.sample_reviews))

if __name__ == '__main__':
    unittest.main() 