import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer
from model import CaptioningTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 128
patch_size = 8
d_model = 192
n_layers = 6
n_heads = 8

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = CaptioningTransformer(
    image_size=image_size,
    in_channels=3,
    vocab_size=tokenizer.vocab_size,
    device=device,
    patch_size=patch_size,
    n_layers=n_layers,
    d_model=d_model,
    n_heads=n_heads,
).to(device)

model_path = "image_captioning_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def make_prediction(
    model, sos_token, eos_token, image, max_len=50, temp=0.5, device=device
):
    log_tokens = [sos_token]
    with torch.inference_mode():
        image_embedding = model.encoder(image.to(device))
        for _ in range(max_len):
            input_tokens = torch.cat(log_tokens, dim=1)
            data_pred = model.decoder(input_tokens.to(device), image_embedding)
            dist = torch.distributions.Categorical(logits=data_pred[:, -1] / temp)
            next_tokens = dist.sample().reshape(1, 1)
            log_tokens.append(next_tokens.cpu())
            if next_tokens.item() == 102:
                break
    return torch.cat(log_tokens, dim=1)


def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    sos_token = 101 * torch.ones(1, 1).long().to(device)
    tokens = make_prediction(model, sos_token, 102, img_tensor)
    caption = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return caption


with gr.Blocks(css=".block-title { font-size: 24px; font-weight: bold; }") as demo:
    gr.Markdown("<div class='block-title'>Image Captioning with PyTorch</div>")
    gr.Markdown("Upload an image and get a descriptive caption about the image:")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Your Image")
            generate_button = gr.Button("Generate Caption")
        with gr.Column():
            caption_output = gr.Textbox(
                label="Caption Output",
                placeholder="Your generated caption will appear here...",
            )

    generate_button.click(fn=predict, inputs=image_input, outputs=caption_output)

if __name__ == "__main__":
    demo.launch(share=True)
