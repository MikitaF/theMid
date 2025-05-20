##Automated Art Generator theMid 
the Mid is a tool that helps creating art assets for the game in an automated manner
using API call to AI services (GPT and Grok to start with)
the tool has a simple web interface consisting of text fiels and buttons

## process and UI
user fill out the following fields
#settings:
    1. API key for GPT (Store in .env file: GPT_API_KEY)
    2. API key for Grok (Store in .env file: GROK_API_KEY)
    3. git Username (Store in .env file: GIT_USERNAME)
    (Note: Other sensitive credentials like Git tokens for automated pushes should also be stored in the .env file.)

#game info
    1. IP_name
    2. IP_description
    3. target_audience


#asset info
    1. (dropdown menu) asset_type: character concept, character portrait, game item, buiding
    2. (dropdown menu) entity: player, friendly npc, enemy
    3. asset_description: manual user input for the asset description (e.g. undead, elfs, humans, coffeeshop customers).
    4. event_name: manual user input like Halloween, Christmas, St.Patrick's Day


#Art Reference
    1. (image upload)sample image: <user selects the image → image copied to the designated folder → image pushed to git repository → the git raw link is created> 
    (Note: Automated git pushes will require appropriate credentials, ideally a Git token, stored securely, e.g., in the .env file as GIT_TOKEN.)


#brainstorm and research
[button: "create an asset"]
(When this button is pressed, a "Generation in progress..." message will be displayed while the automated fields are being populated.)
fields in this block are filled out automatically after the button has been pressed
    1. <this one is filled out automatically by Grok> based on the following prompt with variables in [] being filled out from the text fields completed by the user
        "you are the customer research specialist working on identifying popular trends aligning with the [IP_description] direction and [target_audience] interest. Make a concise list of latest key trends from entertainment media and social media trends. Output in following jason format:
         {
            "IP trends":{
                observations:"",
                "notable Shows and Movies:[],
                "popular characters":[],
                "competitor games":[]
            }
        } "
    2. <this one is filled out automatically by Grok> based on the following prompt with variables in [] being filled out from the text fields completed by the user
        "you are the marketing research specialist working on identifying popular trends aligning with the [IP_description], [asset_description] direction based on [target_audience] interest in the contest of [event_name]. Make a concise list of key trends from entertainment media and social media trends. Output in following jason format:
        {
            "Event trends":{
                observations:"",
                "notable Shows and Movies:[],
                "popular characters":[],
                "competitor games":[]
            }
        } "
    3.  art_style <image(link) is sent to GPT via API to describe the style using the following prompt to fill out the next field>
        prompt: "thoroughly describe the style of the [image] in detail. 
        Output in following json format:
        {
            "style": {
                "description": "",
                "scale":"realistic/cartoony/exaggerated",
                "
            }
            "linework":{
                "outline": true/false,
                "thickness": "very thin/medium/bold",
                "style": "clean and consistent/rough and abrupt",
                "color": "darker shade of fill tone to preserve cohesion"
                },
            "color_palette": {
                "type": "vivid rgb/neon/realistic/narrow palette/gritty/minimalist",
                "saturation": "saturated/grayscale/pastelle",
                "accents":"yes/no",
                "main colours":[],
                },
            "visual_density": {
                "level": "low/mid/high/etremly detailed",
                },
            additional notes: ""
        }"

when info in all 3 brain storm fields has been filled out/updated the script compiles received jasons from 1 and 2 into one (simple merge of the two JSON objects), let's call it research_info
if asset info is assembled successfully it is used in the following brainstorm sequence of prompts:
1. brainstorm. send a requst to GPT vie following prompt:
    "brainstorm a [asset_description] [entity] [asset_type] progression from a [IP_description] game.
    use the information from [research_info] to guid the design decisions
    progression level 1 → level 6
    progression has a continuation and a logical improvement from level to level
    progression has a growing complexity of the silhouette from simple to complex level 1 → level 6
    elements that has been removed in one of the level are not coming back in the following levels
    [asset_type] have a repetition of elements between neighbour levels providing the recognisability and sense of belonging to one collection unified by a subject matter. At the same time progression avoids having boring repetitiveness in silhoutte.

    avoid including parts of the environment

    output as a json of the following structure:
    {
        "concept name:"",
        "level1": {      
            "name": "",
            "description": "",
            "elements": []
        }
        "level2": {      
            "name": "",
            "description": "",
            "elements": []
        }
        "level3": {      
            "name": "",
            "description": "",
            "elements": []
        }
        "level4": {      
            "name": "",
            "description": "",
            "elements": []
        }
        "level5": {      
            "name": "",
            "description": "",
            "elements": []
        }
        "level6": {      
            "name": "",
            "description": "",
            "elements": []
        }
    }

2. feedback the received result sent back to gpt through the following prompt:
    "evaluate the progression desribed in 
    [brainstorm] 
    come up with actionable feedback for each of the levels if necessary
    output as a json:
    {  "evaluation": {
            "level1": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            },
            "level2": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            },            
            "level3": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            },            
            "level4": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            },            
            "level5": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            },
            "level6": {
                "strengths": [],
                "issues": [],
                "suggestions": []
            }
    }"

3. final_result. the received result from feedback being sent to GPT to iterate on initial brainstorm
    "adjust [brainstorm] based on [feedback]


4. [final_result] is displayed to the user in a designated text field as raw JSON.

##asset creation
1. image creation. [final_result] and [art_style] is being used to create an image via prompt:
    "create an image 3:2 aspect ratio
    with a grid consisting of a 6 same size squares
    the grid consists of 2 rows by 3 squares each
    each square has a tiny [level #] test in it's very left top corner
    top row level 1, level 2, level 3
    bottom row level 4, level 5, level 6

    add a [asset_description] [entity] [asset_type] progression from following json structure:
    [final_result]
    
    using the style described in the following json structure:
    [art_style]
    "

2. the image is displayed in a designated part of the web page

## Error Handling and Validation
The system will implement the following error handling and validation mechanisms:
1.  **API Call Errors:**
    *   **Network Issues/Timeouts:** Implement retries with exponential backoff for transient network errors. After a certain number of retries, inform the user of the failure.
    *   **API Key Errors:** If an API key is invalid or has insufficient quota, the system will notify the user to check their settings (and the .env file).
    *   **Rate Limits:** Implement respectful rate limit handling. If a rate limit is hit, the system should pause and retry after the recommended cool-down period. Inform the user if delays occur due to rate limiting.
    *   **Invalid API Responses/Malformed JSON:**
        *   Log the raw response from the API for debugging.
        *   Attempt to parse JSON responses. If parsing fails, notify the user of an issue with the AI's response format.
        *   For critical steps, consider a fallback or asking the user to try again.
2.  **Input Validation:**
    *   **API Keys:** Check for the presence of API keys in the .env file before attempting to make calls.
    *   **Dropdown Selections:** Ensure selections are within the expected range of options (though primarily handled by UI).
    *   **Image Upload:** Validate file type (e.g., JPG, PNG) and potentially size before processing and uploading to Git.
3.  **Git Operations:**
    *   Handle potential errors during git commands (e.g., authentication failure, push conflicts). Notify the user if an image fails to sync with the repository.
4.  **User Feedback:**
    *   Provide clear, user-friendly error messages that guide the user on how to resolve the issue or what to try next. Avoid exposing raw technical error details directly to the user unless in a specific debug mode.
    *   Maintain a log file for more detailed error tracking for development and troubleshooting purposes.