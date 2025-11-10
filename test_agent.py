import os
import asyncio
from dotenv import load_dotenv

# Load environment variables FIRST, before importing agent
load_dotenv()

from agent import run_agent

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "GITHUB_USERNAME", "GITHUB_PERSONAL_ACCESS_TOKEN"]
    missing = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value or value == f"your_{var.lower()}":
            missing.append(var)
    
    if missing:
        print("❌ Missing or placeholder environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set these in your .env file with actual values")
        print("\nYour .env file should look like:")
        print("GOOGLE_API_KEY=AIza...")
        print("GITHUB_USERNAME=your_username")
        print("GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...")
        return False
    
    print("✅ All environment variables are set")
    print(f"   - Google API Key: {os.environ.get('GOOGLE_API_KEY')[:10]}...")
    print(f"   - GitHub Username: {os.environ.get('GITHUB_USERNAME')}")
    print(f"   - GitHub Token: {os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')[:10]}...")
    return True

async def test_simple_script():
    """Test 1: Generate a simple script"""
    print("\n" + "="*60)
    print("TEST 1: Simple Python Script")
    print("="*60)
    
    prompt = """
    Create a Python script that:
    - Reads a text file
    - Counts word frequency
    - Displays top 10 most common words
    """
    
    await run_agent(prompt)

async def test_flask_api():
    """Test 2: Generate a Flask API"""
    print("\n" + "="*60)
    print("TEST 2: Flask REST API")
    print("="*60)
    
    prompt = """
    Create a simple Flask REST API for a todo list application with:
    - GET /todos - list all todos
    - POST /todos - create a new todo
    - PUT /todos/<id> - update a todo
    - DELETE /todos/<id> - delete a todo
    Include proper error handling and use SQLite for storage.
    """
    
    await run_agent(prompt)

async def test_data_processor():
    """Test 3: Generate a data processing script"""
    print("\n" + "="*60)
    print("TEST 3: Data Processing Script")
    print("="*60)
    
    prompt = """
    Create a Python script for CSV data processing that:
    - Reads CSV files using pandas
    - Performs data cleaning (handle missing values)
    - Generates summary statistics
    - Creates visualizations with matplotlib
    - Exports results to Excel
    """
    
    await run_agent(prompt)

async def test_minimal():
    """Test 4: Minimal test - just verify it works"""
    print("\n" + "="*60)
    print("TEST 4: Minimal Test (Quick)")
    print("="*60)
    
    prompt = """
    Create a simple Python script that prints "Hello, World!" with proper documentation.
    """
    
    await run_agent(prompt)

def main():
    """Main test runner"""
    print("🧪 Python Code Builder Agent - Test Suite")
    print("="*60)
    
    # Check environment
    if not check_environment():
        return
    
    # Choose test
    print("\nAvailable tests:")
    print("1. Simple Python Script (word counter)")
    print("2. Flask REST API (todo app)")
    print("3. Data Processing Script (CSV analyzer)")
    print("4. Minimal Test (quick verification)")
    print("5. Custom prompt")
    
    choice = input("\nSelect test (1-5): ").strip()
    
    try:
        if choice == "1":
            asyncio.run(test_simple_script())
        elif choice == "2":
            asyncio.run(test_flask_api())
        elif choice == "3":
            asyncio.run(test_data_processor())
        elif choice == "4":
            asyncio.run(test_minimal())
        elif choice == "5":
            custom_prompt = input("\nEnter your custom prompt: ")
            asyncio.run(run_agent(custom_prompt))
        else:
            print("Invalid choice!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()