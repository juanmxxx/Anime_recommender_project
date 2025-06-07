#!/usr/bin/env python3
"""
Test script to verify the recommendHandler is working correctly
"""

import os
import sys
import traceback

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import recommendHandler
    from AI.recommendHandler import get_recommendations_text, get_recommendations_json
    
    # Test with a simple prompt
    test_prompt = "action anime with magic powers"
    
    print("\n===== TESTING RECOMMENDATION HANDLER =====")
    print(f"Test prompt: '{test_prompt}'")
    
    # Get text recommendations
    print("\n----- TEXT FORMAT -----")
    text_result = get_recommendations_text(test_prompt, top_n=3)
    print(text_result)
    
    # Get JSON recommendations
    print("\n----- JSON FORMAT -----")
    json_result = get_recommendations_json(test_prompt, top_n=3)
    print(json_result)
    
    print("\n===== TEST COMPLETE =====")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
