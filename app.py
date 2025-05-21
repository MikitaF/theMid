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
from io import BytesIO # For in-memory image conversion
from PIL import Image # For image format conversion

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

        elif direct_image_url_input and (direct_image_url_input.startswith("http://") or direct_image_input.startswith("https://")):
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
    "style": {{"description": "", "scale":"realistic/cartoony/exaggerated", "shading": "flat/mostly flat/smooth 3d/detailed realistic/painterly", "design":"angular/curvy/realistic"}},
    "linework":{{"outline": true/false, "thickness": "very thin/medium/bold", "style": "clean and consistent/rough and abrupt", "color": "darker shade of fill tone to preserve cohesion"}},
    "color_palette": {{"type": "vivid rgb/neon/realistic/narrow palette/gritty/minimalist", "saturation": "saturated/grayscale/pastelle", "accents":"yes/no", "main colours":[]}},
    "visual_density": {{"level": "low/mid/high/etremly detailed", "background": "solid color/minimalist/detailed"}},
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
                ip_trends_response = xai_client.chat.completions.create(model="grok-3", messages=[{"role": "user", "content": ip_trends_prompt_text}], response_format={"type": "json_object"}, timeout=30.0)
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
                event_trends_response = xai_client.chat.completions.create(model="grok-3", messages=[{"role": "user", "content": event_trends_prompt_text}], response_format={"type": "json_object"}, timeout=30.0)
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
                    
                    eval_prompt_text_part = f"""You are an art director. You will evaluate a batch of generated images.
For each image, provide a detailed evaluation based on the following criteria.
The images are provided sequentially after this text. Please associate your evaluations with the image IDs listed below.

**Reference Information for Evaluation:**

1.  **Refined Concept**:
    ```json
    {json.dumps(refined_concept_output, indent=2)}
    ```
    Assess how well the image embodies the `concept_name` and `visual_design` described in this Refined Concept.

2.  **Art Style Definition**:
    ```json
    {json.dumps(art_style_info, indent=2)}
    ```
    Assess how well the image adheres to ALL aspects of this Art Style Definition (e.g., style.description, style.scale, linework, color_palette, visual_density). Be specific.

3.  **Prompt Used for Generation (for context only)**:
    `{current_image_prompt_text}`

**Evaluation Criteria per Image**:
-   **Fitness to Refined Concept**: How well does the image embody the `concept_name` and `visual_design` from the Refined Concept?
-   **Adherence to Art Style Definition**: How well does the image match ALL aspects of the specified `Art Style Definition`?
-   **Readability & Composition**: Is the subject clear? Is the composition effective?
-   **Artifacts & Distortions**: Are there any visual glitches, strange anatomy, or other generation artifacts?
-   **Quality Score**: Assign a quality score from 1 (poor) to 10 (excellent). Be critical and use the full range.

**Output Format**:
Output ONLY a single JSON object. The JSON should contain a key "image_evaluations", which is a list of objects. Each object in the list corresponds to one of the evaluated images and MUST include its original 'id'.
The structure for each image evaluation object is:
`{{`
  `"id": "<image_id_from_input_list>",`
  `"fitness_to_concept_evaluation": "<Detailed text assessing how well the image fits the Refined Concept.>",`
  `"adherence_to_art_style_evaluation": "<Detailed text assessing how well the image adheres to ALL aspects of the Art Style Definition.>",`
  `"readability_and_artifacts_evaluation": "<Detailed text on image clarity, composition, and any visual artifacts or distortions.>",`
  `"quality_score": <Integer score from 1 (poor) to 10 (excellent).>,`
  `"overall_feedback": "<Concise summary and actionable suggestions for improving the image or the prompt if generation were to be re-attempted.>"`
`}}`

**List of Image IDs for reference (match with provided images):**
{json.dumps([item['id'] for item in eval_input_for_gpt], indent=2)}
"""
                    # print(f"GPT Image Eval Prompt Text (summary): {eval_prompt_text_part[:500]}...") # Can be very verbose
                    print(f"GPT Image Eval: Evaluating {len(eval_input_for_gpt)} image(s). Refined Concept: '{refined_concept_output.get('concept_name', 'N/A')}'")

                    try:
                        content_parts = [{"type": "text", "text": eval_prompt_text_part}]
                        for item in eval_input_for_gpt: # eval_input_for_gpt has {"id": ..., "url": ...}
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": item["image_url"], "detail": "high"} 
                            })
                        
                        eval_response = gpt_client.chat.completions.create(
                            model="gpt-4o", 
                            messages=[{"role": "user", "content": content_parts}],
                            response_format={"type": "json_object"}, 
                            max_tokens=2500 # Increased max_tokens slightly for potentially more verbose JSON output
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
                current_best_image_info = None
                for img_info in all_generated_images_data:
                    score = img_info.get("evaluation", {}).get("quality_score", 0)
                    if score > best_overall_score:
                        best_overall_score = score
                        current_best_image_info = img_info
                
                if current_best_image_info:
                    final_selected_image_url = current_best_image_info["url"]
                    final_image_evaluation = current_best_image_info["evaluation"]
                    final_selected_image_details_for_stylization = current_best_image_info # CAPTURE THIS
                    print(f"Final selected image: {final_selected_image_url} with score {best_overall_score}. ID: {current_best_image_info.get('id')}")
                elif all_generated_images_data: # If all evals failed or scores were 0
                    final_selected_image_url = all_generated_images_data[-1]["url"] # Pick last one from last batch
                    final_image_evaluation = all_generated_images_data[-1].get("evaluation", {"error": "No valid evaluation found for any image"})
                    final_selected_image_details_for_stylization = all_generated_images_data[-1] # CAPTURE THIS
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

        # --- New Step: Stylize Final Concept Image ---
        stylized_image_prompt_text_gpt = ""
        stylized_image_url_leo = "/static/mock_image.png" # Changed placeholder
        stylized_image_evaluation_gpt = {"error": "Stylization not attempted or failed."}
        stylized_image_overall_error = None # Specific error for this step

        can_attempt_stylization = (
            final_selected_image_details_for_stylization and
            final_selected_image_details_for_stylization.get("url") != "/static/mock_image.png" and
            final_selected_image_details_for_stylization.get("id") and # Crucial: Leonardo ID must exist
            isinstance(art_style_info, dict) and "error" not in art_style_info and
            isinstance(refined_concept_output, dict) and "error" not in refined_concept_output and
            gpt_client and LEO_API_KEY
        )

        if can_attempt_stylization:
            original_leo_image_id = final_selected_image_details_for_stylization.get("id")
            original_image_width = final_selected_image_details_for_stylization.get("width", 896)
            original_image_height = final_selected_image_details_for_stylization.get("height", 896)

            print(f"\n--- Attempting Stylization of Concept Image (ID: {original_leo_image_id}) ---")

            # 1. GPT: Create prompt for DALL·E Image Edit API
            # The original_leo_image_id is the ID from Leonardo. We need its URL for GPT context / DALL·E input.
            concept_image_url_for_stylization = final_selected_image_details_for_stylization.get("url")

            prompt_for_dalle_edit_text = f"""You are an expert prompt engineer for DALL·E's image editing capabilities.
Your goal is to create a prompt that will instruct DALL·E to redraw an existing concept image (which will be provided as input to DALL·E) using a specific art style, while preserving the original image's composition, characters, and key details.

**Original Concept Image URL (for your reference, DALL·E will get image data directly):** {concept_image_url_for_stylization}
Concept Name: {refined_concept_output.get('concept_name', 'N/A')}
Visual Design: {refined_concept_output.get('visual_design', 'N/A')}

**Target Art Style Definition (to be applied):**
{json.dumps(art_style_info, indent=2)}

**Task:**
Generate a concise and effective DALL·E prompt (max 1000 characters). This prompt will be used with DALL·E's image editing function.
The prompt should instruct DALL·E to:
1.  Apply ALL aspects of the 'Target Art Style Definition' to the input image.
2.  Ensure the core subject matter, composition, and essential details (like character poses, main objects, general layout) of the original input image are preserved as much as possible.
3.  The prompt should focus on describing the desired *style transformation* and reinforcing the style elements. 

Output ONLY the DALL·E prompt string for image editing.
Prompt for DALL·E Image Edit:
"""
            print("\n--- GPT: Generating Prompt for DALL·E Image Edit ---")
            dalle_edit_prompt_text_gpt = ""
            try:
                dalle_prompt_gpt_response = gpt_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_for_dalle_edit_text}],
                    max_tokens=250, # DALL-E prompts are shorter
                    temperature=0.3
                )
                dalle_edit_prompt_text_gpt = dalle_prompt_gpt_response.choices[0].message.content.replace("Prompt for DALL·E Image Edit:", "").strip()
                if not dalle_edit_prompt_text_gpt:
                    raise ValueError("GPT returned an empty prompt for DALL·E edit.")
                stylized_image_prompt_text_gpt = dalle_edit_prompt_text_gpt # Store for UI
                print(f"GPT-generated prompt for DALL·E (length {len(dalle_edit_prompt_text_gpt)}):\n{dalle_edit_prompt_text_gpt}")
            except Exception as e:
                stylized_image_overall_error = f"GPT prompt generation for DALL·E edit failed: {e}"
                print(stylized_image_overall_error)

            # 2. Download original concept image for DALL·E input
            image_bytes_for_dalle = None
            if not stylized_image_overall_error and concept_image_url_for_stylization:
                print(f"Downloading concept image from {concept_image_url_for_stylization} for DALL·E input...")
                try:
                    response_img_download = requests.get(concept_image_url_for_stylization, timeout=30)
                    response_img_download.raise_for_status()
                    downloaded_image_bytes = response_img_download.content
                    print("Concept image downloaded successfully.")

                    # Convert downloaded image to PNG in memory
                    print("Converting downloaded image to PNG format for DALL·E...")
                    try:
                        img = Image.open(BytesIO(downloaded_image_bytes))
                        # Ensure image is in RGB or RGBA mode before saving as PNG
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGBA') # Convert to RGBA to handle transparency
                        
                        png_buffer = BytesIO()
                        img.save(png_buffer, format="PNG")
                        image_bytes_for_dalle = png_buffer.getvalue()
                        png_buffer.close()
                        print("Image successfully converted to PNG format.")
                    except Exception as e_convert:
                        stylized_image_overall_error = f"Failed to convert image to PNG for DALL·E: {e_convert}"
                        print(stylized_image_overall_error)

                except Exception as e_download: # Renamed to avoid conflict with e_convert
                    stylized_image_overall_error = f"Failed to download concept image for DALL·E: {e_download}"
                    print(stylized_image_overall_error)
            
            # 3. DALL·E: Generate stylized image using Image Edit API (if prompt and image bytes are ready)
            if not stylized_image_overall_error and dalle_edit_prompt_text_gpt and image_bytes_for_dalle:
                print(f"\n--- DALL·E: Generating Stylized Image via Edit API ---")
                try:
                    # Determine a DALL·E supported size, e.g., 1024x1024
                    # We could try to match original aspect ratio if DALL-E versions support non-square edit outputs well
                    # For DALL-E 2 edit, size must be one of "256x256", "512x512", or "1024x1024"
                    # DALL-E 3 via API might have different constraints or behaviors with gpt-4o driving it.
                    # Let's assume gpt_client.images.edit refers to DALL-E 2 unless API changes noted.
                    target_dalle_size = "1024x1024" 
                    print(f"Requesting DALL·E image edit with size: {target_dalle_size}")

                    response_dalle_edit = gpt_client.images.edit(
                        image=image_bytes_for_dalle,
                        prompt=dalle_edit_prompt_text_gpt,
                        n=1,
                        size=target_dalle_size # e.g., "1024x1024"
                    )
                    if response_dalle_edit.data and response_dalle_edit.data[0].url:
                        stylized_image_url_leo = response_dalle_edit.data[0].url # Reusing variable for UI
                        print(f"DALL·E stylized image generated: {stylized_image_url_leo}")
                    else:
                        stylized_image_overall_error = "DALL·E edit API call succeeded but no image URL found in response."
                        print(stylized_image_overall_error)
                        print(f"Full DALL·E response: {response_dalle_edit}")

                except Exception as e:
                    stylized_image_overall_error = f"DALL·E image edit API call failed: {e}"
                    print(stylized_image_overall_error)
            elif not stylized_image_overall_error: # Catch if we didn't proceed to DALL-E due to earlier failure
                 if not dalle_edit_prompt_text_gpt:
                    stylized_image_overall_error = stylized_image_overall_error or "Skipping DALL·E: Prompt not generated."
                 elif not image_bytes_for_dalle:
                    stylized_image_overall_error = stylized_image_overall_error or "Skipping DALL·E: Original image not downloaded."
                 print(stylized_image_overall_error if stylized_image_overall_error else "Skipping DALL-E for unknown reason before API call.")
            
            # 4. GPT Evaluation of DALL·E stylized image (replaces old Leo eval spot)
            if stylized_image_url_leo and stylized_image_url_leo != "/static/mock_image.png" and not stylized_image_overall_error:
                eval_prompt_stylized_text = f"""You are an art director evaluating a stylized image generated by DALL·E.
The image at [Stylized Image URL] was generated by taking an original concept and attempting to redraw it in a specific 'Target Art Style'.

**Original Concept Details (for context on subject/composition):**
{json.dumps(refined_concept_output, indent=2)}

**Target Art Style (that should have been applied):**
{json.dumps(art_style_info, indent=2)}

**Evaluate the Stylized Image based on these criteria:**
1.  **Adherence to Target Art Style**: How well does the image match ALL aspects of the 'Target Art Style Definition' (e.g., style.description, linework, color_palette, visual_density)? Be specific. Score 1-10.
2.  **Preservation of Original Concept**: How well does the image retain the core subject matter, composition, character poses, and essential details of the 'Original Concept'? Score 1-10.
3.  **Overall Quality & Artifacts**: Note clarity, composition effectiveness (considering the style), and any visual glitches or distortions. Score 1-10.

Output ONLY a single JSON object in the following format:
{{
  "stylized_image_evaluation": {{
    "adherence_to_style_score": <integer_score_1_to_10>,
    "adherence_to_style_comments": "<detailed_text_assessment>",
    "preservation_of_concept_score": <integer_score_1_to_10>,
    "preservation_of_concept_comments": "<detailed_text_assessment>",
    "overall_quality_score": <integer_score_1_to_10>,
    "overall_quality_comments": "<detailed_text_assessment_including_artifacts>"
  }}
}}
"""
                print("\n--- GPT: Evaluating Stylized Image ---")
                try:
                    eval_stylized_response = gpt_client.chat.completions.create(
                        model="gpt-4o", 
                        messages=[{"role": "user", "content": [{"type": "text", "text": eval_prompt_stylized_text}, {"type": "image_url", "image_url": {"url": stylized_image_url_leo, "detail": "high"}}]}],
                        response_format={"type": "json_object"}, 
                        max_tokens=1500 
                    )
                    stylized_image_evaluation_gpt = json.loads(eval_stylized_response.choices[0].message.content)
                    print("Stylized image evaluation successful.")
                except Exception as e:
                    error_msg = f"GPT evaluation of stylized image failed: {e}"
                    stylized_image_evaluation_gpt = {"error": error_msg}
                    print(error_msg)
            elif not stylized_image_overall_error: # If stylize URL is still mock, but no specific error from DALL·E
                stylized_image_overall_error = stylized_image_overall_error or "Stylization process completed but no valid stylized image URL was obtained. Skipping evaluation."
                stylized_image_evaluation_gpt = {"error": stylized_image_overall_error}
                print(stylized_image_overall_error)


        else: # Not can_attempt_stylization
            if not (final_selected_image_details_for_stylization and final_selected_image_details_for_stylization.get("url") != "/static/mock_image.png"):
                stylized_image_overall_error = "Stylization skipped: No valid final concept image was selected or its details are missing."
            elif not (final_selected_image_details_for_stylization and final_selected_image_details_for_stylization.get("id")):
                stylized_image_overall_error = "Stylization skipped: Leonardo ID of the final concept image is missing. Required for image-to-image."
            elif not (isinstance(art_style_info, dict) and "error" not in art_style_info):
                stylized_image_overall_error = "Stylization skipped: Art style information is missing or invalid."
            elif not (isinstance(refined_concept_output, dict) and "error" not in refined_concept_output):
                stylized_image_overall_error = "Stylization skipped: Refined concept information is missing or invalid."
            elif not gpt_client:
                stylized_image_overall_error = "Stylization skipped: GPT client not initialized."
            elif not LEO_API_KEY:
                stylized_image_overall_error = "Stylization skipped: LeonardoAI API key not configured."
            else: # Should not be reached if can_attempt_stylization logic is complete
                stylized_image_overall_error = "Stylization skipped: Prerequisites not met for an unknown reason."
            print(f"Stylization not attempted: {stylized_image_overall_error}")
            stylized_image_evaluation_gpt = {"error": stylized_image_overall_error}


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
            "image_generation_overall_error": image_generation_global_error if final_selected_image_url == "/static/mock_image.png" else None,
            # New data for stylized image
            "stylized_image_prompt_text_gpt": stylized_image_prompt_text_gpt,
            "stylized_image_url_leo": stylized_image_url_leo,
            "stylized_image_evaluation_gpt": stylized_image_evaluation_gpt,
            "stylized_image_overall_error": stylized_image_overall_error
        }
        
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) 