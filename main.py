import gradio as gr
import google.generativeai as genai
import env

genai.configure(api_key=env.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def greet(name, extravagance):
    prompt = f'Greet someone named ${name}. On a scale of 0 to 10, where 0 is the most straightforward and 10 is the most extravagant, you should be at a ${extravagance}.'
    response = gemini_model.generate_content(prompt)
    return response.text

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(), gr.Slider(minimum=0, maximum=10, step=1)],
    outputs=[gr.TextArea()],
    flagging_mode='never'
)

demo.launch()
