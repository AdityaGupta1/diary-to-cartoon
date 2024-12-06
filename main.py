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


# for debugging only
def debug_print(title, contents):
    print()
    print('==================================================')
    print(title)
    print('==================================================')
    print(contents)
    print()


def describe_image(diary_image, author_image):
    if diary_image is None or author_image is None:
        gr.Warning("Please upload both images before generating the comic.")
        return
    # OCR Set Up
    
    progress = gr.Progress()
    progress(0, 'reading text from diary image...')

    ocr_texts, img = ocr_module.get_ocr_texts(diary_image=diary_image)
    diary_contents = ocr_module.validate_and_consolidate_with_gemini(ocr_texts, img)

    debug_print("DIARY CONTENTS", diary_contents)

    progress(0.25, 'generating author description...')

    gemini_prompt = f'''
Here are example descriptions of individuals, based on their images. These examples serve as guidelines for style and structure:

A black American man in his early to mid-30s with a dark complexion, a full, thick black beard, braided hair, and an oval face shape, wearing a blue sports jersey
-----
A white American man in his late 20s to early 30s, with a close-cropped haircut, a short beard, and an oval face shape, wearing glasses and a black hoodie with a small white logo
-----
A Korean woman in her early 20s, with long, wavy reddish-brown hair styled in pigtails, straight bangs, and a round face shape, wearing a white button-up shirt
-----
A Filipino woman in her 60s, with short, slightly wavy dark hair, and a round face shape, wearing a bright pink sweater
-----
A Indian woman in her early 20s, with long, straight black hair, an angular face, and a light blue tank top with thin black straps
-----
A Pakistani man in his 60s, with short, graying hair, a prominent, weathered face with deep wrinkles, and a salt-and-pepper beard, wearing a light-colored collared shirt
-----
A Mexican man in his 30s, with medium brown skin, styled curly hair, and a thin mustache, wearing large red-tinted sunglasses, a retro-style patterned suit with a wide collar, and layered necklaces
-----
An Italian man in his 30s with a light complexion and a neatly trimmed beard, wearing a black fedora hat with a brown band, a dark, textured overcoat with a double-breasted design, and a dark undershirt
-----
A Bengal cat with a sleek coat patterned in gold and black rosettes, and bright green eyes
-----
A vintage typewriter, with a black metal body, round white keys with black lettering, and a chrome return lever
-----
A Golden Retriever puppy, with fluffy cream-colored fur, floppy ears, and a wagging tail
-----
A small bonsai tree, with a thick, gnarled trunk, lush green foliage, and a ceramic pot painted with blue waves
-----
A European Robin, with a bright orange chest, brown wings, and a small, sharp black beak
-----
A classic red and white Coca-Cola vending machine, with a retro design and a shiny chrome handle
-----
A Siamese fighting fish, with iridescent blue and green scales, long flowing fins, and a forked tail

Use these examples to generate a description of the individual in the attached image.
Describe their features such as age, ethnicity, complexion, hair, clothing, clothing color, face shape, or other distinguishing characteristics in a similar style.
Keep the description within one sentence as it will be used to generate comic prompts.
The description should not include any details of the background or objects/characters other than the main subject.
    '''
    response = gemini_model.generate_content([gemini_prompt, author_image])
    author_description = response.text.strip().replace(".", "")

    debug_print("AUTHOR DESCRIPTION", author_description)

    progress(0.5, 'generating image prompts...')

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

    debug_print("PROMPTS", prompts_v1)

    image_prompts = prompts_v1.split('\n-----\n')

    def prompt_editor(prompt):
        return "Single comic book panel of " + prompt[0:1].lower() + prompt[1:]

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

    progress(0.75, 'selecting panels...')

    example_variation_choices = ""
    example_variation_idx = 0
    for i in range(num_panels):
        example_variation_choices += str(example_variation_idx) + ", "
        example_variation_idx = (example_variation_idx + 1) % num_variations_per_panel
    example_variation_choices = example_variation_choices[:-2]

    gemini_prompt = f'''
Attached is a grid of AI-generated images, with {num_panels} rows and {num_variations_per_panel} columns. Each row contains {num_variations_per_panel} variations of a single panel of a {num_panels}-panel comic. The story of the comic is the following:

{diary_contents}

Pick the best combination of {num_panels} panels such that there is exactly one in each category.

A good panel has:
- A clear subject
- Narrative progression from the previous panel and into the next panel
- An art style consistent with the other selected panels

Characteristics to avoid (i.e. to not select panels with) include:
- Extra limbs
- Image noise
- Too much text
- Warped or distorted imagery

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
    
    # Return the images for each gallery
    return comic_image, variation_selections, *variation_galleries, modified_image_prompts_text, diary_contents, *selected_indices, variations_image


def replace_panel(comic_image, variations_image, *selected_indices):
    """
    Replace a specific panel in the comic with a selected variation
    Args:
        comic_image: Current comic image with 4 panels
        variations_image: Image containing all variations
        panel_index: Which panel to replace (0-3)
        variation_index: Which variation to use (0-3)
    """
    print(selected_indices)
    if comic_image is None or variations_image is None:
        return None
    
    # Convert to PIL Image if needed
    if not isinstance(comic_image, Image.Image):
        comic_image = Image.fromarray(comic_image)
    if not isinstance(variations_image, Image.Image):
        variations_image = Image.fromarray(variations_image)
    
    # Create a copy of the comic image
    new_comic = comic_image.copy()
    # Let's begin
    for panel_index in range(4):
        variation_index = int(selected_indices[panel_index])
        print(f"variation_index:{variation_index}")

         
        # Calculate dimensions
        panel_width = comic_image.width // 2
        panel_height = comic_image.height // 2
        
        # Calculate source coordinates in variations image
        src_x = variation_index * panel_width
        src_y = panel_index * panel_height
        src_box = (src_x, src_y, src_x + panel_width, src_y + panel_height)
        
        # Calculate target coordinates in comic image
        target_x = (panel_index % 2) * panel_width
        target_y = (panel_index // 2) * panel_height
        
        # Extract and paste the selected variation
        variation_panel = variations_image.crop(src_box)
        new_comic.paste(variation_panel, (target_x, target_y))
    
    return new_comic

# UI design
# Define the UI elements
demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    # @TODO: Set the title to be the center of the page
    gr.Markdown("# Image To Comic")
    
    with gr.Row():
        diary_input = gr.Image(type='pil', label='Diary Entry')
        author_input = gr.Image(type='pil', label='Diary Author')
    
    with gr.Row():
        submit_btn = gr.Button("Generate Comic", variant="primary")
    
    variation_galleries = [] # Keep track of the gallery
    selected_indices = [] # Keep track of selected indices
    
    with gr.Column():
        comic_output = gr.Image(type='pil', label='Final Comic')
        with gr.Row():
            regenerate_btn = gr.Button("Regenerate All", variant="secondary")
            replace_btn = gr.Button("Replace Panel", variant="primary")
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
            with gr.Row():
                variations_image = gr.Image(type='pil', label='variations image (debug)')
    
    # Hidden state for history
    history = gr.State([])
    
    
    
    # Callbacks and Event handlers
    
    def get_select_index(evt: gr.SelectData, i):
        '''
        Get the index of the selected image in the current gallery
        '''
        
        gr.Warning(f"Selected index {evt.index} for panel {i}.")
        selected_indices[int(i)] = evt.index
        return evt.index, *selected_indices
    
    # Set the callback for each gallery when the selection changes
    for i, gallery in enumerate(variation_galleries):
        # Update the callback to work with gr.State
        selected_index = selected_indices[i]
        gallery.select(
            fn=get_select_index,
            inputs=gr.Number(i, visible=False),
            outputs=[selected_index, *selected_indices], 
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
            *selected_indices,
            variations_image
        ]
    )
    
    regenerate_btn.click(
        fn=describe_image,
        inputs=[diary_input, author_input],
        outputs=[
            comic_output, 
            variation_selections, 
            *variation_galleries, 
            modified_prompts, 
            diary_contents, 
            *selected_indices,
            variations_image
        ]
    )
    
    replace_btn.click(
        fn=replace_panel,
        inputs=[comic_output, variations_image, *selected_indices],
        outputs=[comic_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
