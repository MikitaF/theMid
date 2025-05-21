import os
import git # For Git operations
import uuid # For generating unique filenames
import json # For parsing GPT response
import requests # For LeonardoAI API calls
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename # For secure file handling
from openai import OpenAI # Import OpenAI
import time # For sleep between polling attempts

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
LEO_API_KEY = os.getenv("LEO_API_KEY") # Added for LeonardoAI

# --- Path to your local repository --- 
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

# LeonardoAI Configuration (Placeholder - adjust with actual API details)
LEONARDO_API_ENDPOINT = "https://cloud.leonardo.ai/api/rest/v1/generations" # Common endpoint, might vary
if not LEO_API_KEY:
    print("Warning: LEO_API_KEY not found in .env. LeonardoAI calls will fail.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/create_asset', methods=['POST'])
def create_asset_route():
    """
    Handles the asset creation request based on theMid 1.1 workflow.
    """
    if request.method == 'POST':
        data = request.form.to_dict()
        print("Form data received:", data) 

        sample_image_url = None
        raw_image_link = None # This will store the URL used for GPT, either from upload or direct input
        art_style_info = {}
        ip_trends_data = {}
        event_trends_data = {}

        # --- Handle Sample Image: Prioritize file upload, then direct URL input ---
        uploaded_file = request.files.get('sample_image')
        direct_image_url_input = data.get('sample_image_url_input', '').strip()

        if uploaded_file and uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
            # --- Process File Upload (Save, Git Push, Get Raw URL) ---
            filename = secure_filename(uploaded_file.filename)
            unique_id = uuid.uuid4().hex
            file_extension = filename.rsplit('.', 1)[1].lower()
            new_filename = f"{unique_id}.{file_extension}"
            image_save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            uploaded_file.save(image_save_path)
            sample_image_url = f"/{UPLOAD_FOLDER}/{new_filename}" # For local display if needed
            print(f"Image saved locally to: {image_save_path}")

            try:
                if not GIT_REPO_URL:
                    raise ValueError("GIT_REPO_URL is not set in .env file for image push.")
                repo = git.Repo(REPO_PATH)
                with repo.config_writer() as gw:
                    if not repo.config_reader().has_section('user') or not repo.config_reader().get_value('user', 'name'):
                        gw.set_value("user", "name", GIT_USERNAME or "AutomatedArtBot")
                    if not repo.config_reader().has_section('user') or not repo.config_reader().get_value('user', 'email'):
                        gw.set_value("user", "email", "bot@example.com")
                
                repo_image_path = os.path.join(UPLOAD_FOLDER, new_filename)
                repo.index.add([image_save_path])
                repo.index.commit(f"Add sample image: {new_filename}")
                print(f"Committed image {new_filename} to local Git repo.")

                remote_name = 'origin'
                push_url = GIT_REPO_URL
                if GIT_TOKEN:
                    if GIT_REPO_URL.startswith("https://"):
                        push_url = GIT_REPO_URL.replace("https://", f"https://{GIT_TOKEN}@")
                
                if remote_name not in [r.name for r in repo.remotes]:
                    repo.create_remote(remote_name, push_url)
                else:
                    with repo.remotes[remote_name].config_writer as cw:
                        cw.set("url", push_url)
                
                current_branch = repo.active_branch.name
                origin = repo.remote(name=remote_name)
                print(f"Attempting to push image to {remote_name}/{current_branch}")
                push_info = origin.push(refspec=f'{current_branch}:{current_branch}')
                
                if push_info[0].flags & git.PushInfo.ERROR:
                    raise Exception(f"Git push failed for sample image: {push_info[0].summary}")
                else:
                    print(f"Successfully pushed sample image to {remote_name}/{current_branch}.")
                    base_raw_url = GIT_REPO_URL.replace(".git", "").replace("github.com", "raw.githubusercontent.com")
                    raw_image_link = f"{base_raw_url}/{current_branch}/{repo_image_path}"
                    print(f"Constructed raw image link from upload: {raw_image_link}")

            except Exception as e:
                error_message = f"Error during sample image Git/Push: {e}"
                print(error_message)
                raw_image_link = None # Ensure it's None if git push fails
                art_style_info = {"error": error_message} # Propagate error to art style step

        elif direct_image_url_input and (direct_image_url_input.startswith("http://") or direct_image_url_input.startswith("https://")):
            # --- Use Direct URL Input --- 
            raw_image_link = direct_image_url_input
            sample_image_url = direct_image_url_input # For display purposes, show the URL provided
            print(f"Using direct image URL for art style analysis: {raw_image_link}")
            # Git operations are skipped in this case
        
        else:
            # --- No valid file upload or direct URL --- 
            print("No sample image file uploaded and no valid direct URL provided.")
            # raw_image_link remains None
            # art_style_info will be handled by the next block if raw_image_link is None

        # --- Call GPT to describe Art Style (uses raw_image_link from either upload or direct input) ---
        if gpt_client and raw_image_link:
            art_style_prompt_text = f"""thoroughly describe the style of the image at {raw_image_link} in detail. Focus on the art direction and style instead of the subject itself. Avoide unnecessary flavour adjectives. Output in following json format:
{{
    "style": {{"description": "", "scale":"realistic/cartoony/exaggerated"}},
    "linework":{{"outline": true/false, "thickness": "very thin/medium/bold", "style": "clean and consistent/rough and abrupt", "color": "darker shade of fill tone to preserve cohesion"}},
    "color_palette": {{"type": "vivid rgb/neon/realistic/narrow palette/gritty/minimalist", "saturation": "saturated/grayscale/pastelle", "accents":"yes/no", "main colours":[]}},
    "visual_density": {{"level": "low/mid/high/etremly detailed"}},
    "additional notes": "describe the style in a concise manner professional specification style"
}}
""" # Ensure this prompt is exactly as you need it.
            print(f"\n--- Sending to GPT for Art Style Analysis (Source: {raw_image_link}) ---")
            # print(f"Art Style Prompt:\n{art_style_prompt_text}\n") # Can be verbose
            try:
                gpt_response = gpt_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role":"user","content":[{"type":"text","text":art_style_prompt_text},{"type":"image_url","image_url":{"url":raw_image_link}}]}],
                    response_format={"type": "json_object"}, 
                    max_tokens=1000
                )
                art_style_info = json.loads(gpt_response.choices[0].message.content)
                print("Art style analysis successful.")
            except Exception as e:
                error_message = f"GPT Art Style Analysis failed: {e}"
                print(error_message)
                art_style_info = {"error": error_message}
        elif "error" not in art_style_info: # If no error from Git push but raw_image_link is still None (e.g. no file/URL provided)
            art_style_info = {"error": "Art style analysis skipped: No sample image (file or URL) provided or processed."}
            print(art_style_info["error"]) 
        # If art_style_info already has an error (e.g. from Git push fail), it will persist.

        # --- XAI/Grok API Calls for Research (Same as before) ---
        if xai_client:
            ip_trends_prompt_text = f"""you are the customer research specialist... Output in following jason format: {{"IP trends":{{"observations":"","notable Shows and Movies":[],"popular characters":[],"competitor games":[]}}}}""" # Shortened for brevity
            # (Full prompt text as in original)
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
                ip_trends_response = xai_client.chat.completions.create(model="grok-3", messages=[{"role": "user", "content": ip_trends_prompt_text}], response_format={"type": "json_object"})
                ip_trends_data = json.loads(ip_trends_response.choices[0].message.content)
            except Exception as e:
                print(f"Error XAI IP Trends: {e}")
                ip_trends_data = {"error": f"Failed XAI IP Trends: {e}"}

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
                event_trends_response = xai_client.chat.completions.create(model="grok-3", messages=[{"role": "user", "content": event_trends_prompt_text}], response_format={"type": "json_object"})
                event_trends_data = json.loads(event_trends_response.choices[0].message.content)
            except Exception as e:
                print(f"Error XAI Event Trends: {e}")
                event_trends_data = {"error": f"Failed XAI Event Trends: {e}"}
        else:
            ip_trends_data = {"error": "XAI client not initialized."}
            event_trends_data = {"error": "XAI client not initialized."}
        
        research_info_combined = { "IP_trends_research": ip_trends_data, "Event_trends_research": event_trends_data}
        research_info_json_string = json.dumps(research_info_combined, indent=2)

        # --- Step 1: GPT - Initial Concept & Visual Design (theMid 1.1) ---
        initial_concept_output = {}
        if gpt_client:
            prompt_initial_concept_text = f"""Given the following asset details and research information, generate a unique concept name and a concise visual design description for a [{data.get('asset_description', '')}] [{data.get('entity', '')}] [{data.get('asset_type', '')}] within a [{data.get('ip_description', '')}] game.
Asset Description: {data.get('asset_description', '')}
Entity: {data.get('entity', '')}
Asset Type: {data.get('asset_type', '')}
IP Description: {data.get('ip_description', '')}
Research Information:
{research_info_json_string}
Focus the 'visual_design' on core distinguishing features or thematic elements.
Output ONLY a single JSON object: {{"concept_name": "<Name>", "visual_design": "<Description>"}}"""
            
            print("\n--- GPT: Initial Concept & Visual Design ---")
            print(f"Prompt:\n{prompt_initial_concept_text}\n")
            try:
                response = gpt_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_initial_concept_text}], response_format={"type": "json_object"}, max_tokens=700)
                initial_concept_output = json.loads(response.choices[0].message.content)
            except Exception as e:
                initial_concept_output = {"error": f"GPT Initial Concept failed: {e}"}
        else:
            initial_concept_output = {"error": "GPT client not initialized."}

        # --- Step 2: GPT - Evaluate Initial Concept (theMid 1.1) ---
        concept_feedback_output = {}
        if gpt_client and "error" not in initial_concept_output:
            prompt_feedback_text = f"""Evaluate the proposed concept:
{json.dumps(initial_concept_output, indent=2)}
Provide actionable feedback. Output ONLY a single JSON object: {{"evaluation": {{"strengths": [], "issues": [], "suggestions": []}}}}"""
            print("\n--- GPT: Evaluate Initial Concept ---")
            print(f"Prompt:\n{prompt_feedback_text}\n")
            try:
                response = gpt_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_feedback_text}], response_format={"type": "json_object"}, max_tokens=1000)
                concept_feedback_output = json.loads(response.choices[0].message.content)
            except Exception as e:
                concept_feedback_output = {"error": f"GPT Concept Feedback failed: {e}"}
        elif "error" in initial_concept_output:
            concept_feedback_output = {"error": "Skipped due to error in initial concept.", "prev_error": initial_concept_output["error"]}
        else:
            concept_feedback_output = {"error": "GPT client not initialized."}

        # --- Step 3: GPT - Iterate on Initial Concept (theMid 1.1) ---
        refined_concept_output = {}
        if gpt_client and "error" not in initial_concept_output and "error" not in concept_feedback_output:
            prompt_refine_text = f"""Adjust the [initial_concept] based on the [feedback].
                                    [initial_concept]: {json.dumps(initial_concept_output, indent=2)}
                                    [feedback]: {json.dumps(concept_feedback_output, indent=2)}
                                    Output ONLY the adjusted concept as a single JSON object, same structure as [initial_concept]."""
            print("\n--- GPT: Iterate on Initial Concept ---")
            print(f"Prompt:\n{prompt_refine_text}\n")
            try:
                response = gpt_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_refine_text}], response_format={"type": "json_object"}, max_tokens=700)
                refined_concept_output = json.loads(response.choices[0].message.content)
            except Exception as e:
                refined_concept_output = {"error": f"GPT Refine Concept failed: {e}"}
        elif "error" in initial_concept_output or "error" in concept_feedback_output:
            refined_concept_output = {"error": "Skipped due to error in prior concept/feedback.", "prev_error": initial_concept_output.get("error") or concept_feedback_output.get("error")}
        else:
            refined_concept_output = {"error": "GPT client not initialized."}

        # --- New Step: GPT - Optimize Prompt for LeonardoAI (theMid 1.1) ---
        optimized_leonardo_prompt = ""
        if gpt_client and "error" not in refined_concept_output and "error" not in art_style_info:
            prompt_for_leo_optimization_text = f"""You are an expert prompt engineer for AI image generation services like LeonardoAI, which has a character limit of around 1500 characters for prompts.
Given the following [Concept] and [Art Style] information, synthesize them into a single, concise, and highly effective image generation prompt.
The prompt should be purely descriptive, focusing on keywords, visual details, and artistic direction.
Avoid conversational language, JSON syntax, or any explanatory text.
The generated prompt must be less than 1400 characters.

[Concept]:
{json.dumps(refined_concept_output, indent=2)}

[Art Style]:
{json.dumps(art_style_info, indent=2)}

Output ONLY the refined image generation prompt string. Make it dense with descriptive keywords and artistic direction.
Refined Prompt for LeonardoAI:
"""
            
            print("\n--- GPT: Optimize Prompt for LeonardoAI ---")
            print(f"Prompt for Optimization (summary):\nConcept Name: {refined_concept_output.get('concept_name', 'N/A')}, Art Style Keys: {list(art_style_info.keys()) if isinstance(art_style_info,dict) else 'N/A'}")
            # print(f"Full Prompt for Optimization:\n{prompt_for_leo_optimization_text}\n") # Potentially very long

            try:
                optimization_response = gpt_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_for_leo_optimization_text}],
                    max_tokens=350,  # Max output tokens (approx 1400 chars / 4 chars_per_token)
                    temperature=0.5 # Encourage more focused output
                )
                optimized_leonardo_prompt = optimization_response.choices[0].message.content.replace("Refined Prompt for LeonardoAI:", "").strip()
                
                if len(optimized_leonardo_prompt) > 1450: # A bit of buffer over 1400
                    print(f"Warning: Optimized prompt is long ({len(optimized_leonardo_prompt)} chars). Truncating.")
                    optimized_leonardo_prompt = optimized_leonardo_prompt[:1450]

                if not optimized_leonardo_prompt:
                    raise ValueError("GPT returned an empty optimized prompt.")

                print(f"Optimized LeonardoAI Prompt (length {len(optimized_leonardo_prompt)}):\n{optimized_leonardo_prompt}")

            except Exception as e:
                error_message = f"GPT Optimize Prompt for LeonardoAI failed: {e}"
                print(error_message)
                optimized_leonardo_prompt = "" # Fallback to empty or basic
                # Store this error to be displayed in UI or to halt image gen
                image_generation_global_error = error_message 
        
        elif "error" in refined_concept_output:
            optimized_leonardo_prompt = ""
            image_generation_global_error = f"Skipping LeonardoAI prompt optimization due to error in refined concept: {refined_concept_output.get('error')}"
        elif "error" in art_style_info:
            optimized_leonardo_prompt = ""
            image_generation_global_error = f"Skipping LeonardoAI prompt optimization due to error in art style: {art_style_info.get('error')}"
        else:
            optimized_leonardo_prompt = ""
            image_generation_global_error = "GPT client not initialized. Cannot optimize prompt for LeonardoAI."
        
        # --- Step 4 & 5: LeonardoAI Image Generation & GPT Evaluation Loop (theMid 1.1) ---
        all_generated_images_data = [] 
        final_selected_image_url = "/static/mock_image.png" 
        final_image_evaluation = {"error": "Image generation not run or failed."}
        image_generation_global_error = None # Initialize to None
        # image_generation_global_error might have been set by the optimization step already

        art_style_valid = isinstance(art_style_info, dict) and "error" not in art_style_info # Re-check, though covered by guard above
        refined_concept_valid = isinstance(refined_concept_output, dict) and "error" not in refined_concept_output # Re-check

        # Ensure we have an optimized prompt to proceed with actual image generation
        if not optimized_leonardo_prompt and not image_generation_global_error:
            image_generation_global_error = "LeonardoAI prompt optimization failed to produce a prompt."

        if LEO_API_KEY and art_style_valid and refined_concept_valid and gpt_client and optimized_leonardo_prompt:
            current_image_prompt_text = optimized_leonardo_prompt 

            MAX_ATTEMPTS = 3
            QUALITY_THRESHOLD = 9 
            LEONARDO_MODEL_ID = "b2614463-296c-462a-9586-aafdb8f00e36"
            LEONARDO_API_ENDPOINT_GENERATIONS = "https://cloud.leonardo.ai/api/rest/v1/generations"
            LEONARDO_API_ENDPOINT_GET_GENERATION = "https://cloud.leonardo.ai/api/rest/v1/generations/{generationId}"
            POLLING_INTERVAL_SECONDS = 5
            MAX_POLLING_ATTEMPTS_LEO = 12 # Renamed to avoid conflict if MAX_POLLING_ATTEMPTS is used elsewhere

            for attempt in range(MAX_ATTEMPTS):
                print(f"\n--- LeonardoAI Image Gen Attempt {attempt + 1}/{MAX_ATTEMPTS} ---")
                print(f"Using Prompt (length {len(current_image_prompt_text)}):\n{current_image_prompt_text}")
                
                generated_batch_urls_ids = [] 
                generation_id_leo = None
                
                try:
                    # Step 1: Submit Generation Job to LeonardoAI
                    headers = {
                        "accept": "application/json",
                        "authorization": f"Bearer {LEO_API_KEY}",
                        "content-type": "application/json"
                    }
                    payload = {
                        "height": 896, 
                        "modelId": LEONARDO_MODEL_ID,
                        "num_images": 3,  
                        "presetStyle": "DYNAMIC", 
                        "prompt": current_image_prompt_text,  
                        "width": 896,
                        # "alchemy": False, # Explicitly false or remove if not supported by model
                    } 
                    print(f"LeonardoAI Submit Payload: {json.dumps(payload, indent=2)}")
                    response_submit = requests.post(LEONARDO_API_ENDPOINT_GENERATIONS, json=payload, headers=headers)
                    response_submit.raise_for_status()
                    response_submit_json = response_submit.json()
                    print(f"LeonardoAI Submit Job Response JSON:\n{json.dumps(response_submit_json, indent=2)}")

                    if response_submit_json.get('sdGenerationJob', {}).get('generationId'):
                        generation_id_leo = response_submit_json['sdGenerationJob']['generationId']
                        print(f"LeonardoAI Generation Job submitted! ID: {generation_id_leo}")
                    else:
                        image_generation_global_error = "LeonardoAI submit error: Could not get generationId from response."
                        print(image_generation_global_error)
                        break # Stop this attempt, try next if available or fail

                except requests.exceptions.HTTPError as http_err:
                    image_generation_global_error = f"LeonardoAI Submit HTTP error (Attempt {attempt+1}): {http_err} - Response: {response_submit.text if response_submit else 'No response'}"
                    print(image_generation_global_error)
                    break 
                except requests.exceptions.RequestException as req_err:
                    image_generation_global_error = f"LeonardoAI Submit Request error (Attempt {attempt+1}): {req_err}"
                    print(image_generation_global_error)
                    break
                except Exception as e_submit: # Broader exception for submission step
                    image_generation_global_error = f"LeonardoAI Submit unexpected error (Attempt {attempt+1}): {e_submit}"
                    print(image_generation_global_error)
                    break
                
                if not generation_id_leo:
                    print("Halting image generation for this attempt due to missing Leonardo generation_id.")
                    # image_generation_global_error should be set by now if it was a submission issue
                    continue # Try next major attempt if any, e.g. after prompt refinement

                # Step 2: Poll for Generation Result from LeonardoAI
                print(f"--- Polling LeonardoAI for Result (ID: {generation_id_leo}) ---")
                get_generation_url = LEONARDO_API_ENDPOINT_GET_GENERATION.format(generationId=generation_id_leo)
                
                for poll_attempt in range(MAX_POLLING_ATTEMPTS_LEO):
                    print(f"Polling LeonardoAI attempt {poll_attempt + 1}/{MAX_POLLING_ATTEMPTS_LEO}...")
                    try:
                        response_fetch = requests.get(get_generation_url, headers=headers)
                        response_fetch.raise_for_status()
                        response_fetch_json = response_fetch.json()
                        # print(f"LeonardoAI Fetch Response JSON (Poll Attempt {poll_attempt+1}):\n{json.dumps(response_fetch_json, indent=2)}") # Can be very verbose

                        generation_data = response_fetch_json.get('generations_by_pk')
                        if not generation_data or not isinstance(generation_data, dict):
                            status_leo = "PENDING" # Fallback assumption
                            print(f"Polling response structure unexpected or missing 'generations_by_pk'. Assuming status: {status_leo}")
                        else:
                            status_leo = generation_data.get('status')
                        
                        print(f"LeonardoAI current generation status: {status_leo}")

                        if status_leo == "COMPLETE":
                            print("LeonardoAI Generation COMPLETE!")
                            generated_images_leo = generation_data.get('generated_images')
                            if generated_images_leo and isinstance(generated_images_leo, list):
                                for i, img_item in enumerate(generated_images_leo[:3]): # Process up to 3 images
                                    img_url = img_item.get("url")
                                    img_id = img_item.get("id", uuid.uuid4().hex) # Use API ID or generate one
                                    if img_url:
                                        generated_batch_urls_ids.append({
                                            "url": img_url, 
                                            "id": img_id, 
                                            "prompt_used": current_image_prompt_text, 
                                            "attempt": attempt + 1,
                                            "evaluation": None,
                                            "leonardo_generation_id": generation_id_leo
                                        })
                                print(f"Successfully parsed {len(generated_batch_urls_ids)} image(s) from LeonardoAI.")
                                break # Break from polling loop, images fetched for this attempt
                            else:
                                image_generation_global_error = "LeonardoAI Error: Status COMPLETE but no 'generated_images' array found."
                                print(image_generation_global_error)
                                break # Break polling, as something is wrong with COMPLETE state
                        elif status_leo == "FAILED":
                            image_generation_global_error = "LeonardoAI Generation FAILED."
                            print(image_generation_global_error)
                            break # Break polling, job failed
                        elif status_leo == "PENDING" or status_leo is None:
                            if poll_attempt < MAX_POLLING_ATTEMPTS_LEO - 1:
                                print(f"Waiting {POLLING_INTERVAL_SECONDS}s before next poll...")
                                time.sleep(POLLING_INTERVAL_SECONDS)
                            else:
                                image_generation_global_error = "LeonardoAI max polling attempts reached, generation not complete."
                                print(image_generation_global_error)
                                # Implicitly breaks polling as loop ends
                        else: # Unknown status
                            print(f"LeonardoAI unknown status '{status_leo}'. Assuming PENDING.")
                            if poll_attempt < MAX_POLLING_ATTEMPTS_LEO - 1:
                                time.sleep(POLLING_INTERVAL_SECONDS)
                            else:
                                image_generation_global_error = "LeonardoAI max polling attempts reached with unknown status."
                                print(image_generation_global_error)
                    
                    except requests.exceptions.HTTPError as http_err_poll:
                        print(f"LeonardoAI Polling HTTP error: {http_err_poll}")
                        if poll_attempt < MAX_POLLING_ATTEMPTS_LEO - 1:
                            time.sleep(POLLING_INTERVAL_SECONDS)
                        else:
                            image_generation_global_error = "LeonardoAI max polling attempts after HTTP error."
                            print(image_generation_global_error)
                    except Exception as e_poll:
                        print(f"LeonardoAI Polling unexpected error: {e_poll}")
                        if poll_attempt < MAX_POLLING_ATTEMPTS_LEO - 1:
                            time.sleep(POLLING_INTERVAL_SECONDS)
                        else:
                            image_generation_global_error = "LeonardoAI max polling attempts after unexpected error."
                            print(image_generation_global_error)
                        # break # Optional: break on any poll error, or let it retry
                
                # After polling loop, check if images were actually fetched for this attempt
                if not generated_batch_urls_ids:
                    # If polling finished (or broke early) and we have no images, 
                    # ensure image_generation_global_error reflects the latest polling issue or a generic failure. 
                    if not image_generation_global_error: # If no specific error was set during polling break
                        image_generation_global_error = f"Failed to fetch images from LeonardoAI for attempt {attempt + 1} after polling."
                    print(image_generation_global_error)
                    # No 'break' here for the outer MAX_ATTEMPTS loop, as GPT eval is next
                    # The GPT eval should ideally not run if generated_batch_urls_ids is empty.
                
                # --- GPT Evaluation of Generated Images (theMid 1.1 update) ---
                if gpt_client and generated_batch_urls_ids: # Only evaluate if we have images
                    print(f"\n--- Evaluating {len(generated_batch_urls_ids)} Images from LeonardoAI Attempt {attempt + 1} with GPT ---")
                    eval_input_for_gpt = [{"id": img_data["id"], "image_url": img_data["url"]} for img_data in generated_batch_urls_ids]
                    # Using mock URLs, GPT can't actually see images here. A real URL would be needed.
                    eval_prompt = f"""You are an art director. Evaluate images based on:
Refined Concept: {json.dumps(refined_concept_output, indent=2)}
Art Style: {json.dumps(art_style_info, indent=2)}
Prompt: {current_image_prompt_text}
For each image, assess fit, readability/artifacts, and give a quality score (1-10, be harsh and very judgmental).
Output ONLY JSON: {{"image_evaluations": [{{"id":"<id>", "fitness_description":"<txt>", "readability_artifacts":"<txt>", "quality_score":<int>, "overall_feedback":"<txt>"}}]}}
Images for evaluation: {json.dumps(eval_input_for_gpt, indent=2)}"""
                    print(f"GPT Image Eval Prompt (shortened for log): {eval_prompt[:300]}...")
                    
                    try:
                        # For real image URLs, content would be:
                        # content_parts = [{"type": "text", "text": eval_prompt_text_part}] 
                        # for item in eval_input_for_gpt: content_parts.append({"type":"image_url", "image_url":{"url": item["image_url"]}})
                        # messages=[{"role": "user", "content": content_parts}]
                        # Since URLs are local/mock, GPT won't see them.
                        
                        eval_response = gpt_client.chat.completions.create(
                            model="gpt-4o", 
                            messages=[{"role": "user", "content": eval_prompt}], # Text-only due to mock URLs
                            response_format={"type": "json_object"}, 
                            max_tokens=2000
                        )
                        eval_data_list = json.loads(eval_response.choices[0].message.content).get("image_evaluations", [])
                        
                        # Merge evaluations back into generated_batch_urls_ids
                        temp_evaluated_batch = []
                        for img_data in generated_batch_urls_ids:
                            evaluation = next((e for e in eval_data_list if e.get("id") == img_data["id"]), None)
                            if not evaluation: # Mock eval if GPT fails for an image
                                 evaluation = {"id": img_data["id"], "quality_score": 3, "overall_feedback":"Mocked: GPT eval missing."}
                            img_data["evaluation"] = evaluation
                            temp_evaluated_batch.append(img_data)
                        
                        all_generated_images_data.extend(temp_evaluated_batch) # Add this batch's results

                        # Check if any image in this batch meets threshold
                        best_score_this_attempt = -1
                        for img_d in temp_evaluated_batch:
                            score = img_d.get("evaluation", {}).get("quality_score", 0)
                            if score > best_score_this_attempt: best_score_this_attempt = score
                        
                        if best_score_this_attempt >= QUALITY_THRESHOLD:
                            print(f"Quality threshold met in attempt {attempt+1}. Best score: {best_score_this_attempt}")
                            break # Exit loop, will select best overall later

                        if attempt < MAX_ATTEMPTS - 1: # If not last attempt and threshold not met
                            print("Quality threshold not met. Asking GPT to refine prompt.")
                            # Refine prompt with GPT
                            feedback_summary_for_refine = [{"id":img["id"], "score":img.get("evaluation",{}).get("quality_score"), "feedback":img.get("evaluation",{}).get("overall_feedback")} for img in temp_evaluated_batch]
                            refine_img_prompt_text = f"""Previous image generation attempt had issues.
Previous Prompt: {current_image_prompt_text}
Feedback on generated images: {json.dumps(feedback_summary_for_refine, indent=2)}

Refine the prompt to improve image quality, adherence to concept, and reduce artifacts, while keeping it concise and under 1400 characters.
Output ONLY the new, refined image generation prompt text.
Refined Prompt for LeonardoAI:
"""
                            print(f"GPT Image Prompt Refine Request (shortened): {refine_img_prompt_text[:200]}...")
                            refine_resp = gpt_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":refine_img_prompt_text}], max_tokens=len(current_image_prompt_text)+200)
                            new_prompt = refine_resp.choices[0].message.content.strip()
                            if new_prompt: current_image_prompt_text = new_prompt
                            else: print("GPT did not return refined prompt, reusing old.")
                        else:
                            print("Max attempts reached for image generation.")

                    except Exception as e_eval:
                        image_generation_global_error = f"GPT Image Evaluation failed (Attempt {attempt+1}): {e_eval}"
                        # Add batch to all_generated_images_data with error state for eval
                        for img_d in generated_batch_urls_ids: img_d["evaluation"] = {"error": f"GPT Eval Failed: {e_eval}"}
                        all_generated_images_data.extend(generated_batch_urls_ids)
                        # Potentially break or continue to next attempt with old prompt if eval fails
                        if attempt < MAX_ATTEMPTS - 1:
                            print("Continuing to next attempt with previous prompt due to evaluation error.")
                        else:
                            print("Max attempts reached, and last evaluation failed.")
                            break # Stop if last attempt's eval also failed.
            
            # After loop, select the best image from all_generated_images_data
            if all_generated_images_data:
                best_overall_score = -1
                selected_image_info = None
                for img_info in all_generated_images_data:
                    score = img_info.get("evaluation", {}).get("quality_score", 0)
                    if score > best_overall_score:
                        best_overall_score = score
                        selected_image_info = img_info
                
                if selected_image_info:
                    final_selected_image_url = selected_image_info["url"]
                    final_image_evaluation = selected_image_info["evaluation"]
                    print(f"Final selected image: {final_selected_image_url} with score {best_overall_score}")
                elif all_generated_images_data: # If all evals failed or scores were 0
                    final_selected_image_url = all_generated_images_data[-1]["url"] # Pick last one from last batch
                    final_image_evaluation = all_generated_images_data[-1].get("evaluation", {"error": "No valid evaluation found for any image"})
                    print(f"No image met criteria or had positive score. Selected last generated: {final_selected_image_url}")

            if not all_generated_images_data and not image_generation_global_error:
                image_generation_global_error = "Image generation process completed without producing images."

        elif not LEO_API_KEY: 
            image_generation_global_error = image_generation_global_error or "LeonardoAI Key missing."
        elif not art_style_valid: 
            image_generation_global_error = image_generation_global_error or "Art Style data invalid for image generation."
        elif not refined_concept_valid: 
            image_generation_global_error = image_generation_global_error or "Refined Concept data invalid for image generation."
        elif not gpt_client: 
            image_generation_global_error = image_generation_global_error or "GPT client not init for image eval/prompt optimization."
        elif not optimized_leonardo_prompt: # Catch if optimized_leonardo_prompt is empty and no specific error set it before
             image_generation_global_error = image_generation_global_error or "Optimized prompt for LeonardoAI is missing or empty."


        if image_generation_global_error and final_selected_image_url == "/static/mock_image.png":
            final_image_evaluation = {"error": image_generation_global_error}


        response_data = {
            "research_info_grok_ip": ip_trends_data,
            "research_info_grok_event": event_trends_data,
            "art_style_gpt": art_style_info, 
            "initial_concept_gpt": initial_concept_output,
            "concept_feedback_gpt": concept_feedback_output,
            "refined_concept_gpt": refined_concept_output,
            "uploaded_sample_image_url": sample_image_url, 
            "raw_github_image_link": raw_image_link,
            "all_generated_images_details": all_generated_images_data, # For UI to display all attempts
            "final_selected_image_url": final_selected_image_url,
            "final_image_evaluation": final_image_evaluation,
            "image_generation_overall_error": image_generation_global_error if final_selected_image_url == "/static/mock_image.png" else None
        }
        
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) 