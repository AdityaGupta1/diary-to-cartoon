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
        gr.Warning("Please upload both images before generating the comic.")
        return
    # OCR Set Up
    num_gemini_steps = 3
    current_gemini_step = 0
    
    progress = gr.Progress()
    progress(0.2, 'Reading text from diary image...')

    orc_texts, img = ocr_module.get_ocr_texts(diary_image=diary_image)
    diary_contents = ocr_module.validate_and_consolidate_with_gemini(orc_texts, img)
    
    # for debugging only
    print(diary_contents)

    progress((current_gemini_step, num_gemini_steps), 'generating prompts v1...')
    current_gemini_step += 1

    gemini_prompt = f'''
    Here are example descriptions of individuals, based on their images. These examples serve as guidelines for style and structure:

    A black American man in his early to mid-30s with a dark complexion, a full, thick black beard, braided hair, and an oval face shape, wearing a blue sports jersey
    -----
    A white American man in his late 20s to early 30s, with a close-cropped haircut, a short beard, and an oval face shape, wearing glasses and a black hoodie with a small white logo
    -----
    A Korean woman in her early 20s, with long, wavy reddish-brown hair styled in pigtails, straight bangs, and a round face shape, wearing a white button-up shirt
    -----
    A Filipino woman in her 60s or 70s, with short, slightly wavy dark hair, and a round face shape, wearing a bright pink sweater
    -----
    A Korean woman in her early 20s, with long, straight black hair, a heart-shaped face, and a light blue tank top with thin black straps
    -----
    A Pakistani man in his 60s or 70s, with short, graying hair, a prominent, weathered face with deep wrinkles, and a salt-and-pepper beard, wearing a light-colored collared shirt
    -----
    A Filipino-American man in his 30s, with medium brown skin, styled curly hair, and a thin mustache, wearing large red-tinted sunglasses, a retro-style patterned suit with a wide collar, and layered necklaces
    -----
    An Italian man in his 30s or 40s with a light complexion and a neatly trimmed beard, wearing a black fedora hat with a brown band, a dark, textured overcoat with a double-breasted design, and a dark undershirt

    Use these examples to generate a description of the individual in the attached image.
    Describe their features such as age, ethnicity, complexion, hair, clothing, clothing color, face shape, or other distinguishing characteristics in a similar style.
    Keep the description within one sentence as it will be used to generate comic prompts.
    '''
    response = gemini_model.generate_content([gemini_prompt, author_image])
    author_description = response.text.strip()
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

The diary's author is described as follows: {author_description}. Use this in the prompts to ensure consistency.
Include the author's color or race as well as other distinguishing characteristics to keep the prompts consistent.

Prompts should be written in the third person.
Prompts should be one sentence at most.

Here is an example of good output:

{author_description} eating waffles and drinking orange juice at a table.
-----
{author_description} sitting in a lecture hall, learning about macroeconomics.
-----
{author_description} at a birthday party with his friends, blowing out the candles on a cake.
-----
{author_description} getting drunk with his friends at a karaoke bar.

Each prompt should fully repeat the description of the author.
Prompts should not refer to other prompts.
Assume that the four prompts will be used in four separate contexts.

Do not include any extra text as your response will be used as a string in a Python program and it needs to be concise.
    '''
    response = gemini_model.generate_content([gemini_prompt])
    prompts_v1 = response.text.strip()

    image_prompts = prompts_v1.split('\n-----\n')

    def prompt_editor(prompt):
        return "A comic book panel of " + prompt[0:1].lower() + prompt[1:]

    modified_image_prompts = [prompt_editor(prompt) for prompt in image_prompts]

    num_panels = 4
    num_variations_per_panel = 4

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

    # Initialize selected indices
    selected_indices = [variation_selections[i] for i in range(len(variation_selections))]
     
    # Prepare images for each gallery
    variation_galleries = [[] for _ in range(num_panels)]
    for panel_idx in range(num_panels):
        for variation_idx in range(num_variations_per_panel):
            variation_galleries[panel_idx].append(images[panel_idx * num_variations_per_panel + variation_idx])
            variation_galleries[panel_idx].selected_index = selected_indices[panel_idx]

    # Return the images for each gallery
    return comic_image, variation_selections, *variation_galleries, modified_image_prompts_text, diary_contents, *selected_indices

def regenerate_comic(diary_image, author_image, history):
    """
    Function to handle regeneration of the comic translation
    Returns a new processed result using the same inputs
    """
    if diary_image is None or author_image is None:
        # Error message
        gr.Warning("Please upload both images before regenerating the comic.")
        return None, None, None, None, None, history
    
    # Process the comic again
    comic_image, variation_selections, *gallery_images, modified_image_prompts_text, diary_contents = describe_image(diary_image, author_image)
    
    # Add to history (optional)
    history = history or []
    history.append((comic_image, variation_selections, *gallery_images, modified_image_prompts_text, diary_contents))
    
    return comic_image, variation_selections, *gallery_images, modified_image_prompts_text, diary_contents, history

def replace_panel(comic_image, variation_galleries):
    """
    Replace panels in the comic with selected variations from galleries
    Args:
        comic_image: Current comic image with 4 panels
        gallery1-4: Gallery components containing the selected images
    """
    # If comic image is none, then simply return the comic image
    if comic_image is None:
        return comic_image
    
    # Convert to PIL Image if needed
    if not isinstance(comic_image, Image.Image):
        comic_image = Image.fromarray(comic_image)
    
    # Calculate dimensions of the comic image
    panel_width = comic_image.width // 2
    panel_height = comic_image.height // 2
    
    # Create a copy of the comic image
    new_comic = comic_image.copy()
    
    # Replace panels that have selections
    for panel_index in range(len(variation_galleries)):
        gallery = variation_galleries[panel_index]
             

    return new_comic


# UI Design
# Define the UI elements
demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    # @TODO: Set the title to be the center of the page
    gr.Markdown("# Image To Comic")
    
    with gr.Row():
        diary_input = gr.Image(type='pil', label='diary entry')
        author_input = gr.Image(type='pil', label='diary author')
    
    with gr.Row():
        submit_btn = gr.Button("Generate Comic", variant="primary")
        regenerate_btn = gr.Button("Regenerate All", variant="secondary")
        with gr.Row():
            with gr.Column():
                replace_btn = gr.Button("Replace Panel", variant="primary")
    
    variation_galleries = [] # Keep track of the gallery
    selected_indices = [] # Keep track of selected indices
    
    with gr.Column():
        comic_output = gr.Image(type='pil', label='Final Comic')
        
        # Define the variation galleries section
        with gr.Row():
            
            for i in range(4):
                with gr.Column():
                    selected_index = gr.Number(label=f'Selected Index for Panel {i+1}')
                    selected_indices.append(selected_index)
                    # Set the selected index for the gallery
                    gallery = gr.Gallery(
                        label=f'Panel {i+1} Variations',
                        columns=4,
                        allow_preview=True,
                        show_label=True,
                        selected_index=selected_index
                    )
                    
                    variation_galleries.append(gallery)
                    
                 
        with gr.Accordion("Debug Information", open=False):
            variation_selections = gr.TextArea(label='variation selections (debug)')
            modified_prompts = gr.TextArea(label='modified image prompts (debug)')
            diary_contents = gr.TextArea(label='diary contents')
    
    # Hidden state for history
    history = gr.State([])
    
    
    
    # Callbacks and Event handlers
    
    def get_select_index(evt: gr.SelectData):
        '''
        Get the index of the selected image in the current gallery
        '''
        gr.Warning(f"Selected index: {evt.index}")
        return evt.index
    
    # Set the callback for each gallery when the selection changes
    for i, gallery in enumerate(variation_galleries):
        # Update the callback to work with gr.State
        selected_index = selected_indices[i]
        gallery.select(
            fn=get_select_index,
            inputs=None,
            outputs=selected_index
        )
    # Define Different Click Events 
    # Set up click events
    submit_btn.click(
        fn=describe_image,
        inputs=[diary_input, author_input],
        outputs=[
            comic_output, 
            variation_selections, 
            *variation_galleries, 
            modified_prompts, 
            diary_contents, 
            *selected_indices
        ]
    )
    
    regenerate_btn.click(
        fn=regenerate_comic,
        inputs=[diary_input, author_input, history],
        outputs=[comic_output, variation_selections, *variation_galleries, modified_prompts, diary_contents, history]
    )
    
    replace_btn.click(
        fn=replace_panel,
        inputs=[
            comic_output,
            variation_galleries[0],  # Pass each gallery individually
            variation_galleries[1],
            variation_galleries[2],
            variation_galleries[3],
        ],
        outputs=[comic_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
