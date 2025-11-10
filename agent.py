import os
import json
import re
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

# Define the state schema
class AgentState(TypedDict):
    user_prompt: str
    generated_code: str
    file_structure: dict
    user_wants_github: bool
    github_repo_name: str
    github_repo_url: str
    error: str
    messages: list

# Initialize the Gemini LLM with lazy initialization
def get_llm():
    """Get or create the LLM instance"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in your .env file"
        )
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )


# GitHub MCP Client Manager
class GitHubMCPClient:
    def __init__(self):
        # Get credentials from environment
        self.token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.username = os.environ.get("GITHUB_USERNAME")
        
        # The MCP endpoint from your tips
        self.base_url = "https://api.githubcopilot.com/mcp/"
        
        if not self.token:
            raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not found in .env")
        if not self.username:
            raise ValueError("GITHUB_USERNAME not found in .env")

        # Set up the authentication headers
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.client = None

    async def __aenter__(self):
        # Initialize the async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=self.headers, 
            timeout=30.0
        )
        await self.client.__aenter__()
        
        # Let's test the connection/authentication
        try:
            # This is a simple MCP call to verify credentials
            await self.client.get("github/user")
            print("✅ GitHub MCP Client Authenticated Successfully.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print("❌ GitHub MCP: Authentication failed (401).")
                raise Exception("Authentication failed. Check your GITHUB_PERSONAL_ACCESS_TOKEN.")
            print(f"❌ GitHub MCP: Connection test failed: {e}")
            raise
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close the client session
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def create_repository(self, name: str, description: str = "", private: bool = False):
        """Create a new GitHub repository via MCP"""
        payload = {
            "owner": self.username,  # <-- ADD THIS LINE
            "name": name,
            "description": description,
            "private": private
        }
        # We call the 'github/create_repository' tool on the MCP server
        response = await self.client.post("github/create_repository", json=payload)
        response.raise_for_status() # Raise an exception for 4xx/5xx
        return response.json()
    
    async def push_files(self, repo_name: str, files: dict, branch: str = "main"):
        """Push files to GitHub repository via MCP"""
        payload = {
            "owner": self.username,
            "repo": repo_name,
            "files": files,
            "branch": branch,
            "message": "Initial commit from Python Code Builder Agent"
        }
        # We call the 'github/push_files' tool on the MCP server
        response = await self.client.post("github/push_files", json=payload)
        response.raise_for_status()
        return response.json()

# Node 1: Generate Python Code (IMPROVED VERSION)
async def generate_code_node(state: AgentState) -> AgentState:
    """Generate Python code based on user prompt - with better parsing"""
    
    # Use a two-step approach: first generate code, then create metadata
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python developer. Generate clean, production-ready Python code.

Follow these guidelines:
- Include proper error handling
- Add comprehensive docstrings
- Follow PEP 8 style guide
- Make it production-ready
- Keep it simple and maintainable

Return ONLY the Python code, nothing else."""),
        ("human", "{prompt}")
    ])
    
    llm = get_llm()
    code_chain = code_prompt | llm
    
    try:
        # Step 1: Generate the code
        print("🔧 Generating code...")
        code_response = await code_chain.ainvoke({"prompt": state["user_prompt"]})
        generated_code = code_response.content.strip()
        
        # Remove markdown code blocks if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        # Step 2: Generate metadata
        print("📝 Generating project files...")
        metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given Python code, provide metadata in this EXACT format:

FILENAME: <filename.py>
DESCRIPTION: <one sentence description>
REQUIREMENTS: <comma-separated list of packages, or "none">

Example:
FILENAME: calculator.py
DESCRIPTION: A simple calculator with basic arithmetic operations
REQUIREMENTS: none"""),
            ("human", "Here's the code:\n\n{code}\n\nProvide metadata in the required format.")
        ])
        
        metadata_chain = metadata_prompt | llm
        metadata_response = await metadata_chain.ainvoke({"code": generated_code[:1000]})  # Send first 1000 chars
        metadata = metadata_response.content.strip()
        
        # Parse metadata
        filename = "main.py"  # default
        description = "Generated Python code"  # default
        requirements = []
        
        for line in metadata.split('\n'):
            if line.startswith("FILENAME:"):
                filename = line.split("FILENAME:")[1].strip()
            elif line.startswith("DESCRIPTION:"):
                description = line.split("DESCRIPTION:")[1].strip()
            elif line.startswith("REQUIREMENTS:"):
                req_str = line.split("REQUIREMENTS:")[1].strip()
                if req_str.lower() != "none":
                    requirements = [r.strip() for r in req_str.split(',')]
        
        # Create file structure
        state["generated_code"] = generated_code
        state["file_structure"] = {
            filename: generated_code
        }
        
        # Add requirements.txt if needed
        if requirements:
            state["file_structure"]["requirements.txt"] = "\n".join(requirements)
        
        # Add README
        readme_content = f"""# {filename.replace('.py', '').replace('_', ' ').title()}

## Description
{description}

## Installation
"""
        if requirements:
            readme_content += """
```bash
pip install -r requirements.txt
```
"""
        else:
            readme_content += "No external dependencies required.\n"
        
        readme_content += f"""
## Usage
```bash
python {filename}
```

## Generated by
Python Code Builder Agent powered by Google Gemini
"""
        state["file_structure"]["README.md"] = readme_content
        
        state["messages"].append({
            "role": "assistant",
            "content": f"✅ Code generated successfully!\n\n**Description:** {description}\n\n**Files created:**\n" + 
                      "\n".join([f"- {fname}" for fname in state["file_structure"].keys()])
        })
        
    except Exception as e:
        state["error"] = f"Failed to generate code: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Error generating code: {str(e)}"
        })
        import traceback
        traceback.print_exc()
    
    return state

# Node 2: Ask User for GitHub Push
async def ask_github_push_node(state: AgentState) -> AgentState:
    """Ask user if they want to push to GitHub"""
    
    if state.get("error"):
        return state
    
    state["messages"].append({
        "role": "assistant",
        "content": "Would you like to push this code to GitHub? (yes/no)"
    })
    
    return state

# Node 3: Get User Response
async def get_user_response_node(state: AgentState) -> AgentState:
    """Get user's response about GitHub push"""
    
    print("\n" + "="*60)
    print("Generated Code Preview:")
    print("="*60)
    for filename, content in state["file_structure"].items():
        print(f"\n📄 {filename}:")
        print("-" * 40)
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print("\n" + "="*60)
    user_input = input("Would you like to push this code to GitHub? (yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        state["user_wants_github"] = True
        repo_name = input("Enter repository name: ").strip()
        state["github_repo_name"] = repo_name
    else:
        state["user_wants_github"] = False
        state["messages"].append({
            "role": "assistant",
            "content": "Code generation completed. Files are ready to use locally."
        })
    
    return state

# Node 4: Push to GitHub
async def push_to_github_node(state: AgentState) -> AgentState:
    """Push code to GitHub using MCP"""
    
    if not state.get("user_wants_github") or state.get("error"):
        return state
    
    try:
        async with GitHubMCPClient() as github_client:
            # Create repository
            state["messages"].append({
                "role": "assistant",
                "content": f"Creating GitHub repository: {state['github_repo_name']}..."
            })
            
            repo_result = await github_client.create_repository(
                name=state["github_repo_name"],
                description=f"Generated by Python Code Builder Agent",
                private=False
            )
            
            # Extract repo URL from result
            repo_url = f"https://github.com/{os.environ.get('GITHUB_USERNAME')}/{state['github_repo_name']}"
            state["github_repo_url"] = repo_url
            
            # Push files
            state["messages"].append({
                "role": "assistant",
                "content": "Pushing files to repository..."
            })
            
            await github_client.push_files(
                repo_name=state["github_repo_name"],
                files=state["file_structure"]
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": f"✅ Successfully pushed code to GitHub!\n\n🔗 Repository URL: {repo_url}"
            })
            
    except Exception as e:
        state["error"] = f"GitHub operation failed: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Error pushing to GitHub: {str(e)}"
        })
    
    return state

# Define routing logic
def route_after_user_response(state: AgentState) -> Literal["push_to_github", "end"]:
    """Route based on user's choice"""
    if state.get("user_wants_github"):
        return "push_to_github"
    return "end"

# Build the graph
def create_code_builder_agent():
    """Create the LangGraph agent"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("ask_github_push", ask_github_push_node)
    workflow.add_node("get_user_response", get_user_response_node)
    workflow.add_node("push_to_github", push_to_github_node)
    
    # Add edges
    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "ask_github_push")
    workflow.add_edge("ask_github_push", "get_user_response")
    workflow.add_conditional_edges(
        "get_user_response",
        route_after_user_response,
        {
            "push_to_github": "push_to_github",
            "end": END
        }
    )
    workflow.add_edge("push_to_github", END)
    
    return workflow.compile()

# Main execution function
async def run_agent(user_prompt: str):
    """Run the code builder agent"""
    
    # Initialize state
    initial_state = {
        "user_prompt": user_prompt,
        "generated_code": "",
        "file_structure": {},
        "user_wants_github": False,
        "github_repo_name": "",
        "github_repo_url": "",
        "error": "",
        "messages": []
    }
    
    # Create and run agent
    agent = create_code_builder_agent()
    
    print("🤖 Python Code Builder Agent Starting...")
    print("="*60)
    
    final_state = await agent.ainvoke(initial_state)
    
    # Print conversation history
    print("\n" + "="*60)
    print("Conversation Summary:")
    print("="*60)
    for msg in final_state["messages"]:
        print(f"\n{msg['content']}")
    
    return final_state

# Example usage
if __name__ == "__main__":
    # Environment variables are loaded from .env automatically via load_dotenv()
    
    # Verify environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file")
        exit(1)
    
    user_prompt = """
    Create a simple Flask REST API for a todo list application with the following features:
    - GET /todos - list all todos
    - POST /todos - create a new todo
    - PUT /todos/<id> - update a todo
    - DELETE /todos/<id> - delete a todo
    Include proper error handling and use SQLite for storage.
    """
    
    asyncio.run(run_agent(user_prompt))