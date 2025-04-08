import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer
from model import make_model, get_sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = make_model(
    device=device,
    tokenizer=tokenizer,
    n_layers=4,
    d_model=768,
    num_labels=5,
    n_heads=8,
    dropout=0.1,
    max_length=32,
)
model.to(device)

model_path = "sentiment_analysis_model.pt"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("No pretrained model found. Using randomly initialized weights.")


def predict_sentiment(text):
    sentiment = get_sentiment(text, model, tokenizer, device, max_length=32)
    return sentiment


css_str = """
body { 
    background-color: #f7f7f7; 
}

.title { 
    font-size: 48px; 
    font-weight: bold; 
    text-align: center; 
    margin-top: 20px; 
}

.description { 
    font-size: 20px; 
    text-align: center; 
    argin-bottom: 40px; 
}
"""

with gr.Blocks(css=css_str) as demo:
    gr.Markdown("<div class='title'>Sentiment Diffusion</div>")
    gr.Markdown(
        "<div class='description'>Enter a sentence and see the predicted sentiment.</div>"
    )
    text_input = gr.Textbox(
        label="Enter Text", lines=3, placeholder="Type your review or sentence here..."
    )
    predict_btn = gr.Button("Predict Sentiment")
    output_box = gr.Textbox(label="Predicted Sentiment")
    predict_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output_box)

if __name__ == "__main__":
    demo.launch(share=True)
