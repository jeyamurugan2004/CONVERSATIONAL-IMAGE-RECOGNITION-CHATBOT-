import os 
import gradio as gr 
from PIL import Image 
from transformers import BlipForConditionalGeneration, BlipProcessor 
 
processor = BlipProcessor.from_pretrained("santhoshk3688/generate-xray-report") 
model = BlipForConditionalGeneration.from_pretrained("santhoshk3688/generate-xray-report") 
 
def generate_report(image): 
    """Generate a CXR report based on the image""" 
    inputs = processor( 
        images=image, 
        text="a chest x-ray", 
        return_tensors="pt" 
    ) 
    output = model.generate(**inputs, max_length=512) 
    report = processor.decode(output[0], skip_special_tokens=True) 
    return report 
 
def chat_with_openai(user_message, previous_report): 
    """Chat with OpenAI after receiving the CXR report""" 
    conversation = [ 
        {"role": "system", "content": "You are a helpful medical assistant."}, 
        {"role": "user", "content": f"Here is a medical report: {previous_report}. Now, {user_message}"} 
    ] 
 
    response = client.chat.completions.create( 
        messages=conversation, 
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1000, 
        model=model_name 
    ) 
 
    return response.choices[0].message.content 
 
def process_image_and_chat(image, user_message, chat_history): 
    """Handle the full process of generating report and chatting""" 
    if chat_history is None: 
        chat_history = [] 
 
    report = generate_report(image) 
    chat_history.append({"role": "assistant", "content": report}) 
 
    openai_response = chat_with_openai(user_message, report) 
    chat_history.append({"role": "user", "content": user_message}) 
    chat_history.append({"role": "assistant", "content": openai_response}) 
 
    return chat_history, chat_history 
 
iface = gr.Interface( 
    fn=process_image_and_chat, 
    inputs=[ 
        gr.Image(type="pil", label="Upload X-Ray Image"), 
        gr.Textbox(label="Your Question", placeholder="Ask a question about the report"), 
        gr.State(value=[]), 
    ], 
    outputs=[ 
        gr.Chatbot(label="Chatbot", type='messages'), 
        gr.State(), 
    ],  
    title="Conversational Image Recognition Chatbot", 
    description="Upload an X-ray image and ask a follow-up question to generate a radiology report and chat 
with a medical assistant" 
) 
 
iface.launch() 
import os 
import gradio as gr 
from PIL import Image 
from transformers import BlipForConditionalGeneration, BlipProcessor 
 
processor = BlipProcessor.from_pretrained("santhoshk3688/generate-xray-report") 
model = BlipForConditionalGeneration.from_pretrained("santhoshk3688/generate-xray-report") 
 
def generate_report(image): 
    """Generate a CXR report based on the image""" 
    inputs = processor( 
        images=image, 
        text="a chest x-ray", 
        return_tensors="pt" 
    ) 
    output = model.generate(**inputs, max_length=512) 
    report = processor.decode(output[0], skip_special_tokens=True) 
    return report 
 
def chat_with_openai(user_message, previous_report): 
    """Chat with OpenAI after receiving the CXR report""" 
    conversation = [ 
        {"role": "system", "content": "You are a helpful medical assistant."}, 
        {"role": "user", "content": f"Here is a medical report: {previous_report}. Now, {user_message}"} 
    ] 
 
    response = client.chat.completions.create( 
        messages=conversation, 
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1000, 
        model=model_name 
    ) 
 
    return response.choices[0].message.content 
 
def process_image_and_chat(image, user_message, chat_history): 
    """Handle the full process of generating report and chatting""" 
    if chat_history is None: 
        chat_history = [] 
 
    report = generate_report(image) 
    chat_history.append({"role": "assistant", "content": report}) 
 
    openai_response = chat_with_openai(user_message, report) 
    chat_history.append({"role": "user", "content": user_message}) 
    chat_history.append({"role": "assistant", "content": openai_response}) 
 
    return chat_history, chat_history 
 
iface = gr.Interface( 
    fn=process_image_and_chat, 
    inputs=[ 
        gr.Image(type="pil", label="Upload X-Ray Image"), 
        gr.Textbox(label="Your Question", placeholder="Ask a question about the report"), 
        gr.State(value=[]), 
    ], 
    outputs=[ 
        gr.Chatbot(label="Chatbot", type='messages'), 
        gr.State(), 
    ], 
    title="Conversational Image Recognition Chatbot", 
    description="Upload an X-ray image and ask a follow-up question to generate a radiology report and chat 
with a medical assistant" 
)