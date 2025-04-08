import os
import torch
import spacy
import gradio as gr
from model import make_model, translate_sentence, Vocab
import __main__

__main__.Vocab = Vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizers():
    try:
        spacy_es = spacy.load("es_core_news_sm")
    except OSError:
        os.system("python -m spacy download es_core_news_sm")
        spacy_es = spacy.load("es_core_news_sm")
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    print("Tokenizers loaded.")
    return spacy_es, spacy_en


spacy_es, spacy_en = load_tokenizers()

if os.path.exists("vocab.pt"):
    torch.serialization.add_safe_globals([__main__.Vocab])
    vocab_src, vocab_trg = torch.load("vocab.pt", weights_only=False)
else:
    raise FileNotFoundError(
        "vocab.pt not found. Please build and save the vocabularies first."
    )

model = make_model(
    device,
    vocab_src,
    vocab_trg,
    n_layers=3,
    d_model=512,
    d_ffn=512,
    n_heads=8,
    dropout=0.1,
    max_length=50,
)
model.to(device)

if os.path.exists("translation_model.pt"):
    model.load_state_dict(torch.load("translation_model.pt", map_location=device))
    print("Pretrained model loaded.")
else:
    raise FileNotFoundError(
        "translation_model.pt not found. Please train and save the model first."
    )


def translate(text):
    translation = translate_sentence(
        text, model, vocab_src, vocab_trg, spacy_es, device, max_length=50
    )
    return translation


css_str = """
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 20px;
        text-align: center;
        margin-bottom: 40px;
    }
"""

with gr.Blocks(css=css_str) as demo:
    gr.Markdown("<div class='title'>Spanish-to-English Translator</div>")
    gr.Markdown(
        "<div class='description'>Enter a Spanish sentence below to receive its English translation.</div>"
    )
    with gr.Row():
        txt_input = gr.Textbox(
            label="Enter Spanish sentence", lines=2, placeholder="Ej: ¿Cómo estás?"
        )
    translate_btn = gr.Button("Translate")
    txt_output = gr.Textbox(label="English Translation", lines=2)
    translate_btn.click(fn=translate, inputs=txt_input, outputs=txt_output)

if __name__ == "__main__":
    demo.launch(share=True)
