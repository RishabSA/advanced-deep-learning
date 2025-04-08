import os
import torch
import gradio as gr
from PIL import Image
from model import sample_ddpm_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(text_prompt, mask_upload, guidance_scale):
    """
    text_prompt: A string containing the text prompt.
    mask_upload: Either a PIL image uploaded by the user or None.
    guidance_scale: Float slider for classifier-free guidance strength.
    """
    generated_img = sample_ddpm_inference(
        text_prompt, mask_upload, guidance_scale, device
    )
    return generated_img


css_str = """
body { 
    background-color: #f7f7f7; 
}

.title { 
    font-size: 48px; 
    text-align: center; 
    margin-top: 20px; 
}

.description { 
    font-size: 20px; 
    text-align: center; 
    margin-bottom: 40px; 
}
"""

with gr.Blocks(css=css_str) as demo:
    gr.Markdown("<div class='title'>Conditioned Latent Diffusion with CelebA</div>")
    gr.Markdown(
        "<div class='description'>Enter a text prompt and (optionally) upload a mask image for conditioning; the model will generate an image accordingly.</div>"
    )
    with gr.Row():
        text_input = gr.Textbox(
            label="Text Prompt",
            lines=2,
            placeholder="E.g., 'She is a woman with blond hair, wearing lipstick.'",
        )
        mask_input = gr.Image(
            label="Optional Mask for Conditioning",
            source="upload",
            tool="editor",
            type="pil",
        )
    guidance_slider = gr.Slider(
        1.0, 5.0, value=1.0, step=0.1, label="Classifier-Free Guidance Scale"
    )
    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image", type="pil")

    generate_button.click(
        fn=generate_image,
        inputs=[text_input, mask_input, guidance_slider],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(share=True)
