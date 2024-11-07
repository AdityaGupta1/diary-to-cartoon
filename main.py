import env
import gradio as gr
import google.generativeai as genai
from google.cloud import vision
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

# Importing Tool modules
import ocr_module

# Set up the Vision API using Credential file
client = vision.ImageAnnotatorClient.from_service_account_file(env.GOOGLE_VISION_API_KEY_PATH)


genai.configure(api_key=env.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

def describe_image(diary_image, author_image):
    if diary_image is None or author_image is None:
        # TODO: actual error message
        return
    # OCR Set Up
    num_gemini_steps = 3
    current_gemini_step = 0
    
    progress = gr.Progress()
    progress(0.2, 'Reading text from diary image...')

    orc_texts, img = ocr_module.get_ocr_texts(diary_image=diary_image)
    diary_contents = ocr_module.validate_and_consolidate_with_gemini(orc_texts, img)
    
    print(diary_contents)

    progress((current_gemini_step, num_gemini_steps), 'generating prompts v1...')
    current_gemini_step += 1

    gemini_prompt = f'''
Here is a diary entry:

{diary_contents}

Split the text into four parts which will comprise four panels of a comic, written in the style used to prompt an AI image generation tool. Separate the prompts with -----, like so:

Prompt 1
-----
Prompt 2
-----
Prompt 3
-----
Prompt 4

The diary's author is shown in the attached image, so use that in the prompts.
Include the author's color or race as well as other distinguishing characteristics to keep the prompts consistent.

Prompts should be written in the third person.
Prompts should be one sentence at most.

Here is an example of good output:

A Polynesian man with short black hair, a green tank top, and glasses eating waffles and drinking orange juice at a table.
-----
A Polynesian man with short black hair, a green tank top, and glasses sitting in a lecture hall, learning about macroeconomics.
-----
A Polynesian man with short black hair, a green tank top, and glasses at a birthday party with his friends, blowing out the candles on a cake.
-----
A Polynesian man with short black hair, a green tank top, and glasses getting drunk with his friends at a karaoke bar.

Each prompt should fully repeat the description of the author.
Prompts should not refer to other prompts.
Assume that the four prompts will be used in four separate contexts.

Do not include any extra text as your response will be used as a string in a Python program and it needs to be concise.
    '''
    response = gemini_model.generate_content([gemini_prompt, author_image])
    prompts_v1 = response.text.strip()

    image_prompts = prompts_v1.split('\n-----\n')

    def prompt_editor(prompt):
        return "A comic book panel of " + prompt[0:1].lower() + prompt[1:]

    modified_image_prompts = [prompt_editor(prompt) for prompt in image_prompts]

    num_panels = 4
    num_variations_per_panel = 2

    images = []
    for panel_idx in range(num_panels):
        for variation_idx in range(num_variations_per_panel):
            progress((panel_idx * num_variations_per_panel + variation_idx, num_panels * num_variations_per_panel), 'generating images...')
            image = pipe(prompt=modified_image_prompts[panel_idx], num_inference_steps=2, guidance_scale=0.0).images[0]
            images.append(image)

    image_width, image_height = images[0].size

    variations_image_width = image_width * num_variations_per_panel
    variations_image_height = image_height * num_panels
    variations_image = Image.new('RGB', (variations_image_width, variations_image_height))
    for x in range(num_variations_per_panel):
        for y in range(num_panels):
            variations_image.paste(images[y * num_variations_per_panel + x], (x * image_width, y * image_height))

    progress((current_gemini_step, num_gemini_steps), 'selecting panels...')
    current_gemini_step += 1

    example_variation_choices = ""
    example_variation_idx = 0
    for i in range(num_panels):
        example_variation_choices += str(example_variation_idx) + ", "
        example_variation_idx = (example_variation_idx + 1) % num_variations_per_panel
    example_variation_choices = example_variation_choices[:-2]

    gemini_prompt = f'''
Attached is a grid of AI-generated images, with {num_panels} rows and {num_variations_per_panel} columns. Each row contains {num_variations_per_panel} variations of a single panel of a {num_panels}-panel comic. The story of the comic is the following:

{diary_contents}

Pick the best combination of {num_panels} panels such that there is exactly one in each category. When choosing panels, take the following into account:
- Image quality (e.g. don't choose images with extra limbs or lots of noise)
- Narrative progression
- Consistent art style

Output your choices in a comma-delimited list, with 0 being the left-most image for that row and {num_variations_per_panel - 1} being the right-most image for that row. Do not include any extra text as your response will be used as a string in a Python program and it needs to be concise.
    '''
    response = gemini_model.generate_content([gemini_prompt, variations_image])
    variation_selections_text = response.text.strip()
    variation_selections = [int(idx) for idx in variation_selections_text.split(",")]

    comic_width = image_width * 2
    comic_height = image_height * 2
    comic_image = Image.new('RGB', (comic_width, comic_height))

    panel_indices = []
    for i in range(num_panels):
        panel_indices.append(i * num_variations_per_panel + variation_selections[i])

    comic_image.paste(images[panel_indices[0]], (0, 0))
    comic_image.paste(images[panel_indices[1]], (image_width, 0))
    comic_image.paste(images[panel_indices[2]], (0, image_height))
    comic_image.paste(images[panel_indices[3]], (image_width, image_height))

    modified_image_prompts_text = "\n\n".join(modified_image_prompts)
    return comic_image, variation_selections, variations_image, modified_image_prompts_text, diary_contents

demo = gr.Interface(
    fn = describe_image,
    inputs = [
        gr.Image(type='pil', label='diary entry'),
        gr.Image(type='pil', label='diary author')
    ],
    outputs = [
        gr.Image(type='pil', label='comic'),
        gr.TextArea(label='variation selections (debug)'),
        gr.Image(type='pil', label='variations (debug)'),
        gr.TextArea(label='modified image prompts (debug)'),
        gr.TextArea(label='diary contents')
    ],
    flagging_mode = 'never'
)

demo.launch(share=False)
