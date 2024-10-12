import env
import gradio as gr
import google.generativeai as genai
from diffusers import AutoPipelineForText2Image
import torch

genai.configure(api_key=env.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

sdxl_pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
sdxl_pipeline.to("cuda")

def describe_image(image):
    # if image is None:
    #     return 'No image provided!'

    # gemini_prompt = 'Read the text in the provided image.'
    # response = gemini_model.generate_content([gemini_prompt, image])

    # return response.text

    sdxl_prompt = "a dog with a wizard hat"
    image = sdxl_pipeline(prompt=sdxl_prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image

demo = gr.Interface(
    fn=describe_image,
    inputs=[gr.Image(type='pil')],
    # outputs=[gr.TextArea()],
    outputs=[gr.Image(type='pil')],
    flagging_mode='never'
)

demo.launch()
