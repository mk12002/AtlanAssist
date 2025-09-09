#!/usr/bin/env python3
"""
Comprehensive test script for Atlan Copilot
Tests all core functionality without running the full Streamlit app
"""

import os
import sys
import json
import traceback

def test_environment():
    """Test if all required environment variables and files exist"""
    print("🔍 Testing Environment Setup...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        return False
    
    # Check FAISS index
    if not os.path.exists("faiss_index"):
        print("❌ FAISS index not found - run 'python build_index.py' first")
        return False
    
    # Check sample data
    if not os.path.exists("data/sample_tickets.json"):
        print("❌ Sample tickets file not found")
        return False
    
    print("✅ Environment setup OK")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n📦 Testing Package Imports...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'plotly.express',
        'plotly.graph_objects',
        'langchain',
        'langchain_google_genai',
        'langchain_community',
        'sentence_transformers',
        'faiss'
    ]
    
    for package in required_packages:
        try:
            if package == 'plotly.express':
                import plotly.express as px
            elif package == 'plotly.graph_objects':
                import plotly.graph_objects as go
            else:
                __import__(package)
            print(f"✅ {package}")
        except (ImportError, AttributeError) as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_modules():
    """Test if custom modules work correctly"""
    print("\n🧩 Testing Custom Modules...")
    
    try:
        # Test classification module
        from modules.classification import get_classification_chain, TicketClassification
        chain = get_classification_chain()
        print("✅ Classification module OK")
        
        # Test RAG module
        from modules.rag import get_rag_chain, get_rag_response
        vector_store, llm = get_rag_chain()
        print("✅ RAG module OK")
        
    except Exception as e:
        print(f"❌ Module error: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_loading():
    """Test if data loading works"""
    print("\n📊 Testing Data Loading...")
    
    try:
        # Test ticket loading
        with open("data/sample_tickets.json", "r") as f:
            tickets = json.load(f)
        
        if len(tickets) == 0:
            print("❌ No tickets found")
            return False
        
        # Verify ticket structure
        required_fields = ['id', 'subject', 'body']
        for field in required_fields:
            if field not in tickets[0]:
                print(f"❌ Missing field: {field}")
                return False
        
        print(f"✅ Loaded {len(tickets)} tickets")
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_classification():
    """Test classification functionality"""
    print("\n🤖 Testing AI Classification...")
    
    try:
        from modules.classification import get_classification_chain
        
        chain = get_classification_chain()
        test_text = "How do I set up SSO with Azure AD?"
        
        result = chain.invoke({"ticket_text": test_text})
        
        # Verify result structure
        if not hasattr(result, 'topic_tags') or not hasattr(result, 'sentiment') or not hasattr(result, 'priority'):
            print("❌ Invalid classification result structure")
            return False
        
        print(f"✅ Classification test passed")
        print(f"   Topics: {result.topic_tags}")
        print(f"   Sentiment: {result.sentiment}")
        print(f"   Priority: {result.priority}")
        return True
        
    except Exception as e:
        print(f"❌ Classification error: {e}")
        traceback.print_exc()
        return False

def test_rag():
    """Test RAG pipeline functionality"""
    print("\n🔎 Testing RAG Pipeline...")
    try:
        from modules.rag import get_rag_response
        test_query = "How do I configure column-level lineage in Atlan?"
        answer, sources = get_rag_response(test_query)
        if not answer or not isinstance(answer, str):
            print("❌ RAG did not return a valid answer")
            return False
        if not isinstance(sources, list):
            print("❌ RAG did not return sources as a list")
            return False
        print(f"✅ RAG test passed\n   Answer: {answer[:80]}...\n   Sources: {sources}")
        return True
    except Exception as e:
        print(f"❌ RAG error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Atlan Copilot Integration Tests\n")
    
    tests = [
        test_environment,
        test_imports,
        test_modules,
        test_data_loading,
        test_classification,
        test_rag
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break  # Stop on first failure
    
    print(f"\n📋 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Atlan Copilot is ready!")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
