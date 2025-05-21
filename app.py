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
                            print(f"\n--- Sending to GPT for Art Style Analysis ---")
                            print(f"Art Style Prompt:\n{art_style_prompt_text}\n")
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
            print("\n--- Requesting IP Trends from XAI/Grok ---")
            print(f"IP Trends Prompt:\n{ip_trends_prompt_text}\n")
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
            event_trends_prompt_text = f"""you are the marketing research specialist working on identifying popular trends aligning with the [{data.get('ip_description', '')}], [{data.get('asset_description', '')}] direction based on [{data.get('target_audience', '')}] interest in the context of [{data.get('event_name', '')}]. Make a concise list of key trends from entertainment media and social media trends. Output in following jason format:
            {{
                "Event trends":{{
                    observations:"",
                    "notable Shows and Movies":[],
                    "popular characters":[],
                    "competitor games":[]
                }}
            }} """
            print("\n--- Requesting Event Trends from XAI/Grok ---")
            print(f"Event Trends Prompt:\n{event_trends_prompt_text}\n")
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

        # --- Combine research_info for GPT Brainstorm prompt ---
        research_info_combined = { "IP_trends_research": ip_trends_data, "Event_trends_research": event_trends_data}
        research_info_json_string = json.dumps(research_info_combined, indent=4) 

        # --- GPT Brainstorm - Step 1: Initial Concept & Visual Design ---
        initial_concept_output = {}
        concept_name_from_gpt = data.get('asset_description', 'Unnamed Concept') # Fallback
        visual_design_from_gpt = "Default visual design description." # Fallback

        if gpt_client:
            prompt_step1_text = f"""Given the following asset details and research information, generate a unique concept name and a detailed visual design description for a [{data.get('asset_description', '')}] [{data.get('entity', '')}] [{data.get('asset_type', '')}] within a [{data.get('ip_description', '')}] game.

Asset Description: {data.get('asset_description', '')}
Entity: {data.get('entity', '')}
Asset Type: {data.get('asset_type', '')}
IP Description: {data.get('ip_description', '')}

Research Information:
{research_info_json_string}

Focus the 'visual_design' on 1-3 core distinguishing features or thematic elements that can be clearly evolved across a 6-level progression. Avoid overly detailed lists of minor accessories at this stage.

Output ONLY a single JSON object with the following structure:
{{
    "concept_name": "<Generated Concept Name>",
    "visual_design": "<Focused Visual Design Description highlighting 1-3 core features>"
}}"""
            print("\n--- Requesting Initial Concept & Visual Design from GPT (Step 1) ---")
            print(f"GPT Initial Concept Prompt:\n{prompt_step1_text}\n")
            try:
                response_step1 = gpt_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_step1_text}],
                    response_format={"type": "json_object"},
                    max_tokens=500 # Adjust as needed for name + visual description
                )
                content_step1 = response_step1.choices[0].message.content
                print(f"GPT Initial Concept Raw Response: {content_step1}")
                if content_step1.strip().startswith("```json"):
                    json_block_step1 = content_step1.strip()[7:-3].strip()
                else:
                    json_block_step1 = content_step1
                initial_concept_output = json.loads(json_block_step1)
                concept_name_from_gpt = initial_concept_output.get("concept_name", concept_name_from_gpt)
                visual_design_from_gpt = initial_concept_output.get("visual_design", visual_design_from_gpt)
            except Exception as e:
                print(f"Error in GPT Brainstorm Step 1 (Initial Concept): {e}")
                initial_concept_output = {"error": f"Failed GPT Brainstorm Step 1: {e}"}
        else:
            print("GPT client not initialized. Skipping GPT Brainstorm Step 1.")
            initial_concept_output = {"error": "GPT client not initialized for Step 1."}

        # --- GPT Brainstorm - Step 2: Progression Brainstorm ---
        brainstorm_output = {} # Initialize
        if gpt_client and "error" not in initial_concept_output:
            prompt_step2_text = f"""Brainstorm a 6-level progression for the concept named '[{concept_name_from_gpt}]'.
The asset is a [{data.get('asset_description', '')}] [{data.get('entity', '')}] [{data.get('asset_type', '')}] from a [{data.get('ip_description', '')}] game.

Adhere to the following visual design constraints for the progression:
{visual_design_from_gpt}

Progression Rules:
- Progression must go from level 1 to level 6.
- Progression should show logical improvement and continuation from one level to the next.
- Complexity of the silhouette must grow from simple (level 1) to complex (level 6).
- Elements removed in one level must not reappear in subsequent levels.
- The [{data.get('asset_type', '')}] should have repeating elements between neighboring levels for recognizability and a sense of belonging to one unified collection, while avoiding boring silhouette repetitiveness.
- Avoid including parts of the environment.

Output ONLY a single JSON object with the following structure (ensure 'concept_name' is '[{concept_name_from_gpt}]'):
{{
    "concept_name": "{concept_name_from_gpt}",
    "level1": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }},
    "level2": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }},
    "level3": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }},
    "level4": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }},
    "level5": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }},
    "level6": {{ "name": "", "description": "", "elements": ["<1-2 key distinguishing elements>"] }}
}}"""
            print("\n--- Requesting Progression Brainstorm from GPT (Step 2) ---")
            print(f"GPT Progression Prompt:\n{prompt_step2_text}\n")
            try:
                brainstorm_response = gpt_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": prompt_step2_text}],
                    max_tokens=2500 # Increased for 6-level detail
                )
                brainstorm_content = brainstorm_response.choices[0].message.content
                print(f"GPT Progression Raw Response: {brainstorm_content}")
                if brainstorm_content.strip().startswith("```json"):
                    json_block_brainstorm = brainstorm_content.strip()[7:-3].strip()
                else:
                    json_block_brainstorm = brainstorm_content
                brainstorm_output = json.loads(json_block_brainstorm)
            except Exception as e:
                print(f"Error calling GPT for Progression Brainstorm (Step 2) or parsing JSON: {e}")
                brainstorm_output = {"error": f"Failed to get Progression Brainstorm from GPT (Step 2): {e}", "concept_name_used": concept_name_from_gpt}
        elif "error" in initial_concept_output:
            brainstorm_output = {"error": "Skipping Progression Brainstorm due to error in Step 1.", "step1_error": initial_concept_output.get("error")}
        else:
            print("GPT client not initialized. Skipping GPT Progression Brainstorm call (Step 2).")
            brainstorm_output = {"error": "GPT client not initialized for Step 2."}

        # --- GPT Feedback on Brainstorm ---
        feedback_output = {}
        if gpt_client and "error" not in brainstorm_output:
            brainstorm_json_string_for_feedback = json.dumps(brainstorm_output, indent=4)
            feedback_prompt_text = f"""evaluate the progression desribed in 
{brainstorm_json_string_for_feedback}
come up with actionable feedback for each of the levels if necessary
output as a json:
{{
    "evaluation": {{
        "level1": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }},
        "level2": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }},
        "level3": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }},
        "level4": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }},
        "level5": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }},
        "level6": {{
            "strengths": [],
            "issues": [],
            "suggestions": []
        }}
    }}
}}"""
            print("\n--- Requesting Feedback on Brainstorm from GPT ---")
            print(f"GPT Feedback Prompt:\n{feedback_prompt_text}\n")
            try:
                feedback_response = gpt_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": feedback_prompt_text}],
                    max_tokens=1500 # Adjust as needed
                )
                feedback_content = feedback_response.choices[0].message.content
                print(f"GPT Feedback Raw Response: {feedback_content}")
                if feedback_content.strip().startswith("```json"):
                    json_block_feedback = feedback_content.strip()[7:-3].strip()
                else:
                    json_block_feedback = feedback_content
                feedback_output = json.loads(json_block_feedback)
            except Exception as e:
                print(f"Error calling GPT for Feedback or parsing JSON: {e}")
                feedback_output = {"error": f"Failed to get Feedback from GPT: {e}"}
        elif "error" in brainstorm_output:
            feedback_output = {"error": "Skipping Feedback due to error in Brainstorm step.", "brainstorm_error": brainstorm_output.get("error")}
        else:
            print("GPT client not initialized. Skipping GPT Feedback call.")
            feedback_output = {"error": "GPT client not initialized for Feedback."}

        # --- GPT Final Result (Iteration on Brainstorm based on Feedback) ---
        final_result_output = {}
        if gpt_client and "error" not in brainstorm_output and "error" not in feedback_output:
            brainstorm_json_for_final = json.dumps(brainstorm_output, indent=4)
            feedback_json_for_final = json.dumps(feedback_output, indent=4)
            final_result_prompt_text = f"""Adjust the [brainstorm] based on the [feedback].

[brainstorm]:
{brainstorm_json_for_final}

[feedback]:
{feedback_json_for_final}

Output ONLY the adjusted brainstorm as a single JSON object, maintaining the exact same structure as the original brainstorm input (concept_name, level1 to level6 with name, description, elements)."""
            print("\n--- Requesting Final Result (Iteration) from GPT ---")
            print(f"GPT Final Result Prompt:\n{final_result_prompt_text}\n")
            try:
                final_result_response = gpt_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": final_result_prompt_text}],
                    max_tokens=2500 # Should be similar to brainstorm output size
                )
                final_result_content = final_result_response.choices[0].message.content
                print(f"GPT Final Result Raw Response: {final_result_content}")
                if final_result_content.strip().startswith("```json"):
                    json_block_final = final_result_content.strip()[7:-3].strip()
                else:
                    json_block_final = final_result_content
                final_result_output = json.loads(json_block_final)
            except Exception as e:
                print(f"Error calling GPT for Final Result or parsing JSON: {e}")
                final_result_output = {"error": f"Failed to get Final Result from GPT: {e}", "original_brainstorm": brainstorm_output}
        elif "error" in brainstorm_output:
            final_result_output = {"error": "Skipping Final Result due to error in Brainstorm step.", "brainstorm_error": brainstorm_output.get("error")}
        elif "error" in feedback_output:
            final_result_output = {"error": "Skipping Final Result due to error in Feedback step.", "feedback_error": feedback_output.get("error"), "original_brainstorm": brainstorm_output}
        else:
            print("GPT client not initialized. Skipping GPT Final Result call.")
            final_result_output = {"error": "GPT client not initialized for Final Result."}
        
        # Fallback for final_result_output if it's empty or an error, to ensure concept_name is present for UI
        if not final_result_output or "error" in final_result_output:
            if "error" in brainstorm_output: # If brainstorm itself had an error
                 final_result_output = {"concept_name": concept_name_from_gpt + " (Error in brainstorm)", **brainstorm_output}
            else: # If brainstorm was okay, but feedback or final iteration failed
                 current_concept_name = brainstorm_output.get("concept_name", concept_name_from_gpt) if isinstance(brainstorm_output, dict) else concept_name_from_gpt
                 final_result_output = brainstorm_output.copy() if isinstance(brainstorm_output, dict) else {"concept_name": current_concept_name}
                 
                 if "error" not in final_result_output: # If brainstorm_output was copied and was not an error itself
                    final_result_output["concept_name"] = current_concept_name + " (Feedback/Final step failed or used brainstorm)"
                 else: # If brainstorm_output was an error or not a dict, ensure a concept_name is present
                    final_result_output["concept_name"] = current_concept_name + " (Error in prior steps)"
                 # Ensure the error from previous step is preserved if it exists
                 if "error" in feedback_output and "error" not in final_result_output:
                     final_result_output["error_from_feedback"] = feedback_output["error"]
                 elif "error" in final_result_output and "error" in feedback_output :
                     final_result_output["error"] = f"{final_result_output.get('error','prior error')} AND {feedback_output.get('error','feedback error')}"

        # --- GPT - Image Creation (DALL-E) ---
        generated_asset_image_url = "/static/mock_image.png" # Default/fallback
        image_generation_error = None

        final_result_valid = isinstance(final_result_output, dict) and "error" not in final_result_output
        art_style_valid = isinstance(art_style_info, dict) and "error" not in art_style_info

        if gpt_client and final_result_valid and art_style_valid:
            try:
                concept_name = final_result_output.get("concept_name", "Unnamed Concept")
                
                # Get asset details from form data for the new prompt
                asset_description_from_form = data.get('asset_description', 'N/A')
                entity_from_form = data.get('entity', 'N/A')
                asset_type_from_form = data.get('asset_type', 'N/A')

                # Convert final_result and art_style to JSON strings for the prompt
                # Using compact representation by specifying separators
                # --- Process final_result_output to simplify for DALL-E ---
                processed_final_result_for_dalle = {}
                if isinstance(final_result_output, dict) and "error" not in final_result_output:
                    processed_final_result_for_dalle["concept_name"] = final_result_output.get("concept_name", "Unnamed Concept")
                    for i in range(1, 7): # Levels 1 to 6
                        level_key = f"level{i}"
                        original_level_data = final_result_output.get(level_key)
                        if isinstance(original_level_data, dict):
                            simplified_desc = original_level_data.get("description", "")
                            # Truncate description to first sentence or ~120 chars
                            if "." in simplified_desc:
                                simplified_desc = simplified_desc.split(".")[0] + "."
                            if len(simplified_desc) > 120:
                                simplified_desc = simplified_desc[:117] + "..."
                            
                            # Ensure elements are simple strings and limit to 2-3 impactful ones if many
                            original_elements = original_level_data.get("elements", [])
                            simplified_elements = [str(el)[:50] for el in original_elements[:3]] # Take first 3, truncate length

                            processed_final_result_for_dalle[level_key] = {
                                "name": original_level_data.get("name", f"Level {i}"),
                                "description": simplified_desc,
                                "elements": simplified_elements
                            }
                        else:
                            # Handle case where a level might be missing or not a dict
                            processed_final_result_for_dalle[level_key] = {"name": f"Level {i} data missing", "description": "", "elements": []}
                else:
                    # If final_result_output itself is an error or not a dict, pass it as is or a placeholder
                    processed_final_result_for_dalle = final_result_output 

                final_result_json_string = json.dumps(processed_final_result_for_dalle, separators=(',', ':'))
                art_style_json_string = json.dumps(art_style_info, separators=(',', ':'))

                image_prompt_text = f"""create an image 3:2 aspect ratio
with a grid consisting of a 6 same size squares
the grid consists of 2 rows by 3 squares each
each square has a tiny [level #] text in it's very left top corner
top row level 1, level 2, level 3
bottom row level 4, level 5, level 6

add a {asset_description_from_form} {entity_from_form} {asset_type_from_form} progression having a separate concept in every square
the concepts should the progression from a very basic initial level to fully evolved subject in level 6.
design notes:
{final_result_json_string}
"""
                # The detailed breakdown like formatted_art_style and levels_prompt_part is replaced by the above.
                
                print("\n--- Requesting Image Grid from DALL-E (User-defined Prompt) ---")
                print(f"DALL-E Image Grid Prompt:\n{image_prompt_text}\n")

                image_response = gpt_client.images.generate(
                    model="dall-e-3",
                    prompt=image_prompt_text,
                    n=1,
                    size="1792x1024", # Landscape aspect ratio suitable for a 2x3 grid
                    quality="standard", # Use "hd" for higher detail if preferred and budget allows
                    # style="vivid" # or "natural" - DALL-E 3 specific
                )
                generated_asset_image_url = image_response.data[0].url
                print(f"DALL-E generated image URL: {generated_asset_image_url}")

            except Exception as e:
                error_message = f"Failed to generate image with DALL-E: {str(e)}"
                print(f"Error during DALL-E image generation: {error_message}")
                image_generation_error = error_message
                # generated_asset_image_url remains "/static/mock_image.png" (set by default)
        
        elif not gpt_client:
            image_generation_error = "GPT client not initialized. Skipping DALL-E call."
        elif not final_result_valid:
            error_detail = final_result_output.get('error', 'Unknown error') if isinstance(final_result_output, dict) else 'Data is not a dictionary'
            image_generation_error = f"Skipping DALL-E call due to error/invalid format in final result: {error_detail}"
        elif not art_style_valid:
            error_detail = art_style_info.get('error', 'Unknown error') if isinstance(art_style_info, dict) else 'Data is not a dictionary'
            image_generation_error = f"Skipping DALL-E call due to error/invalid format in art style: {error_detail}"
        
        if image_generation_error:
            print(f"Image Generation Status: {image_generation_error}")

        mock_generated_asset_image_url = "/static/mock_image.png"

        response_data = {
            "research_info_grok_ip": ip_trends_data,
            "research_info_grok_event": event_trends_data,
            "art_style_gpt": art_style_info, 
            "initial_concept_gpt": initial_concept_output, # Adding step 1 output for visibility
            "brainstorm_gpt": brainstorm_output, # This is now step 2 output
            "feedback_gpt": feedback_output,
            "final_result_gpt": final_result_output,
            "uploaded_sample_image_url": sample_image_url, 
            "raw_github_image_link": raw_image_link, 
            "generated_image_url": generated_asset_image_url,
            "image_generation_error": image_generation_error
        }
        
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) 