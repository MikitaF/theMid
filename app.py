import os
import git # For Git operations
import uuid # For generating unique filenames
import json # For parsing GPT response
# import requests # No longer strictly needed for XAI if using OpenAI library interface
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename # For secure file handling
from openai import OpenAI # Import OpenAI

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load API keys from environment variables
GPT_API_KEY = os.getenv("GPT_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY") 
GIT_USERNAME = os.getenv("GIT_USERNAME")
GIT_TOKEN = os.getenv("GIT_TOKEN")

# --- Path to your local repository --- 
# This should be the root of your git project. 
# For now, we assume the Flask app is run from the root of the repo.
REPO_PATH = os.getcwd() 

# --- GitHub repository URL (replace with your actual repo URL) ---
# Example: "https://github.com/YourUsername/YourRepositoryName.git"
# This is needed to construct the raw file URL and for the push remote.
GIT_REPO_URL = os.getenv("GIT_REPO_URL") # Read from .env

# Initialize OpenAI client for GPT
if not GPT_API_KEY:
    print("Warning: GPT_API_KEY not found in .env. GPT calls will fail.")
    gpt_client = None
else:
    gpt_client = OpenAI(api_key=GPT_API_KEY)

# Initialize OpenAI client for XAI/Grok
if not XAI_API_KEY:
    print("Warning: XAI_API_KEY not found in .env. XAI/Grok calls will fail.")
    xai_client = None
else:
    try:
        xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1" # As per XAI documentation
        )
        print("XAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing XAI client: {e}")
        xai_client = None

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/create_asset', methods=['POST'])
def create_asset_route():
    """
    Handles the asset creation request.
    """
    if request.method == 'POST':
        data = request.form.to_dict()
        print("Form data received:", data) 

        sample_image_url = None
        raw_image_link = None
        art_style_info = {} # Initialize art_style_info
        ip_trends_data = {}
        event_trends_data = {}

        if 'sample_image' in request.files:
            file = request.files['sample_image']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Create a unique filename to avoid overwrites and issues with special chars
                unique_id = uuid.uuid4().hex
                file_extension = filename.rsplit('.', 1)[1].lower()
                new_filename = f"{unique_id}.{file_extension}"
                
                image_save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                file.save(image_save_path)
                sample_image_url = f"/{UPLOAD_FOLDER}/{new_filename}" # Relative path for local display if needed
                print(f"Image saved locally to: {image_save_path}")

                # --- Git Operations ---
                try:
                    if not GIT_REPO_URL:
                        raise ValueError("GIT_REPO_URL is not set in .env file. Cannot push image.")
                    if not GIT_TOKEN:
                        print("Warning: GIT_TOKEN not set. Push might fail if auth is required and not cached.")

                    repo = git.Repo(REPO_PATH)
                    
                    # Configure Git user an email (important for commits if not globally set)
                    # Best to set this globally on your system, but can be set per repo too.
                    # For scripting, it's often safer to ensure it's set.
                    with repo.config_writer() as gw:
                        if not repo.config_reader().has_section('user') or not repo.config_reader().get_value('user', 'name'):
                            gw.set_value("user", "name", GIT_USERNAME or "AutomatedArtBot")
                        if not repo.config_reader().has_section('user') or not repo.config_reader().get_value('user', 'email'):
                            gw.set_value("user", "email", "bot@example.com") # Replace with a generic or your email

                    # Relative path of the image within the repository structure
                    repo_image_path = os.path.join(UPLOAD_FOLDER, new_filename)
                    
                    repo.index.add([image_save_path])
                    commit_message = f"Add sample image: {new_filename}"
                    repo.index.commit(commit_message)
                    print(f"Committed image {new_filename} to local Git repo.")

                    # --- Pushing to remote ---
                    # Ensure you have a remote named 'origin' or adjust as needed
                    # The URL for the remote should include the GIT_TOKEN for authentication
                    # e.g., https://<GIT_TOKEN>@github.com/username/repo.git
                    remote_name = 'origin'
                    if remote_name not in [r.name for r in repo.remotes]:
                        # If remote 'origin' doesn't exist, try to create it
                        # This is a simplified setup. Robust remote handling might need more.
                        print(f"Remote '{remote_name}' not found. Attempting to add it.")
                        # Construct remote URL with token for auth if GIT_TOKEN is present
                        push_url = GIT_REPO_URL
                        if GIT_TOKEN and GIT_USERNAME:
                             # Standard format for token auth is typically just the token, not username
                             # https://<token>@github.com/user/repo.git
                            if GIT_REPO_URL.startswith("https://"):
                                push_url = GIT_REPO_URL.replace("https://", f"https://{GIT_TOKEN}@")
                            else:
                                print("Warning: GIT_REPO_URL is not an HTTPS URL. Token auth might not work as expected.")
                        
                        repo.create_remote(remote_name, push_url)
                        print(f"Added remote '{remote_name}' with URL: {push_url}")
                    else:
                        # If remote exists, update its URL to include the token for the push
                        # This ensures the push uses the token for authentication
                        push_url = GIT_REPO_URL
                        if GIT_TOKEN and GIT_USERNAME:
                            if GIT_REPO_URL.startswith("https://"):
                                push_url = GIT_REPO_URL.replace("https://", f"https://{GIT_TOKEN}@")
                        with repo.remotes[remote_name].config_writer as cw:
                            cw.set("url", push_url)
                        print(f"Updated remote '{remote_name}' URL for push with token.")


                    # Determine the current branch
                    current_branch = repo.active_branch.name
                    origin = repo.remote(name=remote_name)
                    print(f"Attempting to push to {remote_name}/{current_branch}")
                    push_info = origin.push(refspec=f'{current_branch}:{current_branch}')
                    
                    if push_info[0].flags & git.PushInfo.ERROR:
                        print(f"Error during Git push: {push_info[0].summary}")
                        # Potentially include more detailed error info if available
                        # print(f"Details: {push_info[0].error}") 
                        raise Exception(f"Git push failed: {push_info[0].summary}")
                    else:
                        print(f"Successfully pushed to {remote_name}/{current_branch}.")
                        # Construct raw image link (assuming GitHub and main/master branch)
                        # Example: https://raw.githubusercontent.com/YourUsername/YourRepo/main/uploads/image.png
                        base_raw_url = GIT_REPO_URL.replace(".git", "").replace("github.com", "raw.githubusercontent.com")
                        # Try to get the default branch name, common ones are main or master
                        # This might need to be more robust if you use other branch names
                        default_branch = current_branch # Use current branch after successful push
                        raw_image_link = f"{base_raw_url}/{default_branch}/{repo_image_path}"
                        print(f"Constructed raw image link: {raw_image_link}")

                        # --- Call GPT to describe Art Style ---
                        if gpt_client and raw_image_link and not raw_image_link.startswith("Error"):
                            print(f"Sending image to GPT for style analysis: {raw_image_link}")
                            art_style_prompt_text = f'''thoroughly describe the style of the image at {raw_image_link} in detail. Output in following json format: ... (your full JSON prompt format here) ... ''' # Truncated for brevity, ensure full prompt is here
                            # Ensure your full JSON structure is in the prompt like before
                            art_style_prompt_text = f"""thoroughly describe the style of the image at {raw_image_link} in detail. 
Output in following json format:
{{
    "style": {{
        "description": "",
        "scale":"realistic/cartoony/exaggerated"
    }},
    "linework":{{
        "outline": true/false,
        "thickness": "very thin/medium/bold",
        "style": "clean and consistent/rough and abrupt",
        "color": "darker shade of fill tone to preserve cohesion"
    }},
    "color_palette": {{
        "type": "vivid rgb/neon/realistic/narrow palette/gritty/minimalist",
        "saturation": "saturated/grayscale/pastelle",
        "accents":"yes/no",
        "main colours":[]
    }},
    "visual_density": {{
        "level": "low/mid/high/etremly detailed"
    }},
    "additional notes": ""
}}"""
                            gpt_response = gpt_client.chat.completions.create(
                                model="gpt-4o", messages=[{"role":"user","content":[{"type":"text","text":art_style_prompt_text},{"type":"image_url","image_url":{"url":raw_image_link}}]}] , max_tokens=1000
                            )
                            gpt_content = gpt_response.choices[0].message.content
                            if gpt_content.strip().startswith("```json"): json_block = gpt_content.strip()[7:-3].strip()
                            else: json_block = gpt_content
                            art_style_info = json.loads(json_block)
                        elif not gpt_client:
                            art_style_info = {"error": "GPT client not initialized. Check API key."}
                        else:
                            art_style_info = {"error": "Could not get raw image link to send to GPT."}

                except git.GitCommandError as e:
                    print(f"Git command error: {e}")
                    # Fallback or error message for UI
                    raw_image_link = f"Error during Git operation: {e}"
                    art_style_info = {"error": f"Art style analysis skipped due to Git error: {e}"}
                except ValueError as e:
                    print(f"Configuration error: {e}")
                    raw_image_link = f"Git configuration error: {e}"
                    art_style_info = {"error": f"Art style analysis skipped due to Git config error: {e}"}
                except Exception as e:
                    print(f"An unexpected error occurred during Git/GPT operations: {e}")
                    raw_image_link = f"Unexpected Git/GPT error: {e}"
                    art_style_info = {"error": f"Art style analysis skipped due to unexpected error: {e}"}
            else:
                if file.filename == '':
                    print("No image file selected.")
                else:
                    print(f"File type not allowed: {file.filename}")
        # --- Placeholder for Brainstorm & Research Logic ---
        # In future phases, this is where calls to Grok/XAI and GPT will happen.
        
        # --- XAI/Grok API Calls for Research --- 
        if xai_client:
            # 1. IP Trends Research
            ip_trends_prompt_text = f"""you are the customer research specialist working on identifying popular trends aligning with the [{data.get('ip_description', '')}] direction and [{data.get('target_audience', '')}] interest. Make a concise list of latest key trends from entertainment media and social media trends. Output in following jason format:
            {{
                "IP trends":{{
                    observations:"",
                    "notable Shows and Movies":[],
                    "popular characters":[],
                    "competitor games":[]
                }}
            }} """
            print("Requesting IP Trends from XAI/Grok...")
            try:
                ip_trends_response = xai_client.chat.completions.create(
                    model="grok-3", # As per XAI documentation
                    messages=[{"role": "user", "content": ip_trends_prompt_text}],
                    # temperature=0.7 # Optional: add if needed
                )
                ip_trends_content = ip_trends_response.choices[0].message.content
                print(f"XAI IP Trends Raw Response: {ip_trends_content}")
                ip_trends_data = json.loads(ip_trends_content) # Assuming Grok directly returns the JSON string for these prompts
            except Exception as e:
                print(f"Error calling XAI for IP Trends or parsing JSON: {e}")
                ip_trends_data = {"error": f"Failed to get IP Trends from XAI/Grok: {e}"}

            # 2. Event Trends Research
            event_trends_prompt_text = f"""you are the marketing research specialist working on identifying popular trends aligning with the [{data.get('ip_description', '')}], [{data.get('description', '')}] direction based on [{data.get('target_audience', '')}] interest in the context of [{data.get('event_name', '')}]. Make a concise list of key trends from entertainment media and social media trends. Output in following jason format:
            {{
                "Event trends":{{
                    observations:"",
                    "notable Shows and Movies":[],
                    "popular characters":[],
                    "competitor games":[]
                }}
            }} """
            print("Requesting Event Trends from XAI/Grok...")
            try:
                event_trends_response = xai_client.chat.completions.create(
                    model="grok-3", # As per XAI documentation
                    messages=[{"role": "user", "content": event_trends_prompt_text}]
                )
                event_trends_content = event_trends_response.choices[0].message.content
                print(f"XAI Event Trends Raw Response: {event_trends_content}")
                event_trends_data = json.loads(event_trends_content) # Assuming Grok directly returns the JSON string
            except Exception as e:
                print(f"Error calling XAI for Event Trends or parsing JSON: {e}")
                event_trends_data = {"error": f"Failed to get Event Trends from XAI/Grok: {e}"}
        else:
            print("XAI client not initialized. Skipping XAI/Grok calls.")
            ip_trends_data = {"error": "XAI client not initialized. Check API key."}
            event_trends_data = {"error": "XAI client not initialized. Check API key."}

        # --- Mocked data for remaining steps (will be replaced in future phases) ---
        brainstorm_output = {
            "concept_name": f"Mock Concept for {data.get('description', 'N/A')}",
            "level1": {"name": "Level 1 Mock", "description": "Desc 1", "elements": ["Elem A"]},
            "level6": {"name": "Level 6 Mock", "description": "Desc 6", "elements": ["Elem F"]}
        }
        feedback_output = {
            "evaluation": {
                f"level{i+1}": {
                    "strengths": [f"Mock strength for level {i+1}"],
                    "issues": [],
                    "suggestions": [f"Mock suggestion for level {i+1}"]
                } for i in range(6)
            }
        }
        final_result_output = brainstorm_output 
        final_result_output["concept_name"] += " (Adjusted)"
        mock_generated_asset_image_url = "/static/mock_image.png" 

        response_data = {
            "research_info_grok_ip": ip_trends_data, # Now uses actual XAI response or error
            "research_info_grok_event": event_trends_data, # Now uses actual XAI response or error
            "art_style_gpt": art_style_info, 
            "brainstorm_gpt": brainstorm_output,
            "feedback_gpt": feedback_output,
            "final_result_gpt": final_result_output,
            "uploaded_sample_image_url": sample_image_url, 
            "raw_github_image_link": raw_image_link, 
            "generated_image_url": mock_generated_asset_image_url 
        }
        
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) 