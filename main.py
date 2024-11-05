import env
import gradio as gr
import google.generativeai as genai
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

num_panels = 4

genai.configure(api_key=env.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

def describe_image(diary_image, author_image):
    if diary_image is None or author_image is None:
        # TODO: actual error message
        return

    num_gemini_steps = 3
    current_gemini_step = 0
    progress = gr.Progress()
    progress((current_gemini_step, num_gemini_steps), 'reading diary...')
    current_gemini_step += 1

    gemini_prompt = '''
Read the text in the image and output it. The image is of a diary entry, so it may be written in cursive or some other fancy script.

Remove any references to copyrighted material and replace them with some generic text.

Do not include any extra text, as your response will be used verbatim in a Python program for further processing.
    '''
    response = gemini_model.generate_content([gemini_prompt, diary_image])
    diary_contents = response.text

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

Here is an example:

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
    prompts_v1 = response.text

    progress((current_gemini_step, num_gemini_steps), 'generating prompts v2...')
    current_gemini_step += 1

    gemini_prompt = f'''
Here are some prompts for use with an image generation tool:

{prompts_v1}

Rewrite the prompts to ensure that each one contains an independent description of the main subject. Additionally, for any recurring characters, make up some description of them and use that same description in each prompt in which that character appears.

For example:

A corgi with white and brown fur eating waffles and drinking orange juice at a table.
-----
The same corgi sitting in a lecture hall, learning about macroeconomics.
-----
The same corgi at a birthday party with his friends, blowing out the candles on a cake.
-----
The same corgi getting drunk with his friends at a karaoke bar.

should turn into:

A corgi with white and brown fur eating waffles and drinking orange juice at a table.
-----
A corgi with white and brown fur sitting in a lecture hall, learning about macroeconomics.
-----
A corgi with white and brown fur at a birthday party with an orange tabby cat and a green turtle, blowing out the candles on a cake.
-----
A corgi with white and brown fur getting drunk with an orange tabby cat and a green turtle at a karaoke bar.

Keep the same format as the input prompts. That is, use ----- to separate the prompts and keep prompts concise.
Prompts should be at most 1 sentence each.

Do not include any extra text as your response will be used as a string in a Python program and it needs to be concise.
    '''
    response = gemini_model.generate_content(gemini_prompt)
    prompts_v2 = response.text

    progress2 = gr.Progress()

    image_prompts = prompts_v2.split('\n-----\n')

    image_prompt_suffix = 'cartoon style, soft rounded edges, expressive characters, simple shading'

    images = []
    for i in range(num_panels):
        progress2((i, num_panels), 'generating images...')
        modified_image_prompt = image_prompts[i] + ' ' + image_prompt_suffix
        image = pipe(prompt=modified_image_prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        images.append(image)

    width, height = images[0].size

    new_width = width * 2
    new_height = height * 2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (width, 0))
    new_image.paste(images[2], (0, height))
    new_image.paste(images[3], (width, height))

    return new_image, prompts_v2, prompts_v1, diary_contents

demo = gr.Interface(
    fn = describe_image,
    inputs = [
        gr.Image(type='pil', label='diary entry'),
        gr.Image(type='pil', label='diary author')
    ],
    outputs = [
        gr.Image(type='pil', label='comic'),
        gr.TextArea(label='prompts v2 (debug)'),
        gr.TextArea(label='prompts v1 (debug)'),
        gr.TextArea(label='diary contents')
    ],
    flagging_mode = 'never'
)

demo.launch(share=False)
