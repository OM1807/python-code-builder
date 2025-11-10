import os
import json
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

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

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # or "gemini-1.5-flash" for faster responses
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True  # Gemini handles system messages differently
)

# GitHub MCP Client Manager
class GitHubMCPClient:
    def __init__(self, config_path="mcp_config.json"):
        self.config_path = config_path
        self.session = None
        
    async def __aenter__(self):
        # Load MCP configuration
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        github_config = config['mcpServers']['github']
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=github_config['command'],
            args=github_config['args'],
            env=github_config.get('env', {})
        )
        
        # Initialize MCP client
        stdio_transport = await stdio_client(server_params).__aenter__()
        self.stdio_context = stdio_transport
        self.session = ClientSession(stdio_transport[0], stdio_transport[1])
        await self.session.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(exc_type, exc_val, exc_tb)
    
    async def create_repository(self, name: str, description: str = "", private: bool = False):
        """Create a new GitHub repository"""
        result = await self.session.call_tool(
            "create_repository",
            arguments={
                "name": name,
                "description": description,
                "private": private
            }
        )
        return result
    
    async def push_files(self, repo_name: str, files: dict, branch: str = "main"):
        """Push files to GitHub repository"""
        result = await self.session.call_tool(
            "push_files",
            arguments={
                "owner": os.environ.get("GITHUB_USERNAME"),
                "repo": repo_name,
                "files": files,
                "branch": branch,
                "message": "Initial commit from Python Code Builder Agent"
            }
        )
        return result

# Node 1: Generate Python Code
async def generate_code_node(state: AgentState) -> AgentState:
    """Generate Python code based on user prompt"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python developer. Generate clean, well-documented Python code based on user requirements.

CRITICAL: You must return a valid JSON object with the following structure:
{{
    "main_file": "filename.py",
    "code": "<FULL_CODE_HERE>",
    "additional_files": {{
        "requirements.txt": "package1\\npackage2",
        "README.md": "# Project Title\\n\\nDescription..."
    }},
    "description": "Brief description of what the code does"
}}

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown, no code blocks, no backticks
2. The "code" field must contain the COMPLETE Python code as a single string
3. Use \\n for newlines in the code string
4. Escape all quotes inside the code with \\
5. Make the code production-ready with error handling and documentation
6. Do not wrap the JSON in ```json or ``` markers

Example of correct format:
{{"main_file": "app.py", "code": "def hello():\\n    print('Hello')", "description": "Simple hello function"}}"""),
        ("human", "{prompt}")
    ])
    
    
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({"prompt": state["user_prompt"]})
        
        # Parse the response
        content = response.content
        
        # Clean up the response - remove markdown code blocks if present
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        code_data = json.loads(json_str)
        
        state["generated_code"] = code_data["code"]
        state["file_structure"] = {
            code_data["main_file"]: code_data["code"],
            **code_data.get("additional_files", {})
        }
        state["messages"].append({
            "role": "assistant",
            "content": f"✅ Code generated successfully!\n\n**Description:** {code_data['description']}\n\n**Files created:**\n" + 
                      "\n".join([f"- {fname}" for fname in state["file_structure"].keys()])
        })
    except json.JSONDecodeError as e:
        state["error"] = f"Failed to parse JSON response: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Error parsing code generation response: {str(e)}\nResponse was: {content[:200]}"
        })
    except Exception as e:
        state["error"] = f"Failed to generate code: {str(e)}"
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Error generating code: {str(e)}"
        })
    
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
    
    # In a real implementation, this would wait for user input
    # For now, we'll simulate it
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
    # Make sure your .env file contains:
    # GOOGLE_API_KEY=your_api_key_here
    # GITHUB_USERNAME=your_github_username
    # GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here
    
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