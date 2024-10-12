# models/responder.py

from models.model_loader import load_model
from transformers import GenerationConfig
from dotenv import load_dotenv
from logger import get_logger
from openai import OpenAI
from PIL import Image
import torch
import base64
import os
import io

logger = get_logger(__name__)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(images, query, session_id, resized_height=280, resized_width=280, model_choice='qwen', answer_length='short'):
    """
    Generates a response using the selected model based on the query and images.
    """
    try:
        logger.info(f"Generating response using model '{model_choice}' with answer length '{answer_length}'.")
        
        # Convert resized_height and resized_width to integers
        resized_height = int(resized_height)
        resized_width = int(resized_width)
        
        # Ensure images are full paths
        full_image_paths = [os.path.join('static', img) if not img.startswith('static') else img for img in images]
        
        # Check if any valid images exist
        valid_images = [img for img in full_image_paths if os.path.exists(img)]
        
        if not valid_images:
            logger.warning("No valid images found for analysis.")
            return "No images could be loaded for analysis."
        
    

        if model_choice == 'qwen':
            from qwen_vl_utils import process_vision_info
            # Load cached model
            model, processor, device = load_model('qwen')
            # Ensure dimensions are multiples of 28
            resized_height = (resized_height // 28) * 28
            resized_width = (resized_width // 28) * 28

            image_contents = []
            for image in valid_images:
                image_contents.append({
                    "type": "image",
                    "image": image,  # Use the full path
                    "resized_height": resized_height,
                    "resized_width": resized_width
                })
            messages = [
                {
                    "role": "user",
                    "content": image_contents + [{"type": "text", "text": query}],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            # Determine max_new_tokens based on answer_length
            if answer_length == 'short':
                max_new_tokens = 128
            elif answer_length == 'long':
                max_new_tokens = 500
            else:
                max_new_tokens = 128  # Default to short if invalid value
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            logger.info("Response generated using Qwen model.")
            return output_text[0]
        
        elif model_choice == 'gpt4':
            api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key =api_key
            
            
            
            # Determine max_tokens based on answer_length
            max_tokens = 128 if answer_length == 'short' else 500

            # Prepare the messages content with text and images
            content = [
                {"type": "text", "text": query}
            ]

            # Add images to the content
            for img_path in images:
                with open(img_path, "rb") as img_file:
                    image_bytes = img_file.read()
                    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                    content.append({
                        "type": "image",
                        "image": {
                            "bytes": encoded_image
                        }
                    })

            messages = [
                {"role": "user", "content": content}
            ]

            # Send the request to OpenAI API
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )

            generated_text = response['choices'][0]['message']['content']
            logger.info("Response generated using GPT-4 with Vision model.")
            return generated_text
            
        else:
            logger.error(f"Invalid model choice: {model_choice}")
            return "Invalid model selected."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred while generating the response: {str(e)}"
