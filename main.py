import env
import gradio as gr
import google.generativeai as genai
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

genai.configure(api_key=env.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

sdxl_pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
sdxl_pipeline.to("cuda")

def describe_image(diary_image, author_image):
    if diary_image is None or author_image is None:
        return 'No image provided!'

    gemini_prompt = '''
Read the text in the first image. Split the text into four parts which will comprise four panels of a comic, written in the style used to prompt an AI image generation tool. Separate the prompts with -----, like so:

Prompt 1
-----
Prompt 2
-----
Prompt 3
-----
Prompt 4

Prompts should be written in the third person. The diary's author is shown in the second image, so use that in the prompts. Prompts should be one sentence at most. Here is an example:

A corgi with white and brown fur eating waffles and drinking orange juice at a table.
-----
A corgi with white and brown fur sitting in a lecture hall, learning about macroeconomics.
-----
A corgi with white and brown fur at a birthday party with his friends, blowing out the candles on a cake.
-----
A corgi with white and brown fur getting drunk with his friends at a karaoke bar.

Each prompt should be independent and should fully repeat the description of the author.
    '''
    response = gemini_model.generate_content([gemini_prompt, diary_image, author_image])

    image_prompts = response.text.split('\n-----\n')

    image_prompt_suffix = 'cartoon style, soft rounded edges, expressive characters, simple shading'

    images = []
    for image_prompt in image_prompts:
        modified_image_prompt = image_prompt + ' ' + image_prompt_suffix
        image = sdxl_pipeline(prompt=modified_image_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        images.append(image)

    width, height = images[0].size

    new_width = width * 2
    new_height = height * 2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (width, 0))
    new_image.paste(images[2], (0, height))
    new_image.paste(images[3], (width, height))

    return new_image, response.text

demo = gr.Interface(
    fn=describe_image,
    inputs=[gr.Image(type='pil', label='diary entry'), gr.Image(type='pil', label='diary author')],
    outputs=[gr.Image(type='pil', label='comic'), gr.TextArea(label='prompts (debug)')],
    flagging_mode='never'
)

demo.launch()
