import os
from openai import OpenAI
from dotenv import load_dotenv

def test_dalle_generation():
    """
    Tests DALL-E image generation with a simple prompt.
    """
    load_dotenv()
    gpt_api_key = os.getenv("GPT_API_KEY")

    if not gpt_api_key:
        print("Error: GPT_API_KEY not found in .env file.")
        return

    try:
        client = OpenAI(api_key=gpt_api_key)
        print("OpenAI client initialized.")
        
        prompt_text = f"""create an image 3:2 aspect ratio
                        with a grid consisting of a 6 same size squares
                        the grid consists of 2 rows by 3 squares each
                        each square has a tiny [level #] test in it's very left top corner
                        top row level 1, level 2, level 3
                        bottom row level 4, level 5, level 6

                        add an [orc warrior] character progression having a separate concept in the first 5 squares
                        the concepts should the progression from a very basic initial level to fully upgraded character in level 5. the 6th square doesn't have a concept in it

                        all [orc warrior] concepts fully fit into the squares and are not cropped
                    """
        print(f"Attempting to generate image with prompt: \"{prompt_text}\"")

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            n=1,
            size="1024x1024",  # DALL-E 3 supports 1024x1024, 1792x1024, or 1024x1792
            quality="standard" 
        )
        
        image_url = response.data[0].url
        print(f"Successfully generated image!")
        print(f"Image URL: {image_url}")

    except Exception as e:
        print(f"An error occurred during DALL-E test: {e}")

if __name__ == "__main__":
    test_dalle_generation() 