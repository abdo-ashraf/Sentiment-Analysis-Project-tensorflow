import torch
import gradio as gr
from Vocabulary import spacy_tokenizer
from Model_define import Sentiment_LSTM
import os

os.system('python -m spacy download en_core_web_sm')

device = 'gpu' if torch.cuda.is_available() else 'cpu'

model = torch.load("lstm_model.bin", map_location=device, weights_only=False)
model_state = torch.load("lstm_model_states.pt", map_location=device, weights_only=False)


vocab = model_state['vocabulary']
tokenizer = spacy_tokenizer()

cls_to_idx = model_state['class_dict']
idx_to_cls = {value:key for key,value in cls_to_idx.items()}

def pre_processor(text):
    tokens = tokenizer(text.lower())
    unk_id = vocab.get('<UNK>', None)
    return torch.tensor([vocab.get(word, unk_id) for word in tokens])

def post_processor(raw_output):
    label = (raw_output >= 0.5).int()
    return idx_to_cls[label.item()].capitalize(), round(raw_output.item(), 2)


@torch.no_grad
def lunch(raw_input):
    input = pre_processor(raw_input)
    output = model(input.unsqueeze(0), device)
    return post_processor(output)

custom_css ='.gr-button {background-color: #bf4b04; color: white;}'

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label='Input a Review or click an Example')
            gr.Examples(["It is no wonder that the film has such a high rating, it is quite literally breathtaking. What can I say that hasn't said before? Not much, it's the story, the acting, the premise, but most of all, this movie is about how it makes you feel. Sometimes you watch a film, and can't remember it days later, this film loves with you, once you've seen it, you don't forget.",
                          "This film is nothing but one cliche after another. Having seen many of the 100's of prison films made from the early 30's to the 50's, I was able to pull almost every minute of Shawcrap from one of those films."],
                        inputs=input_text, label="Examples: ")
        with gr.Column():
            class_name = gr.Textbox(label="This review is")
            confidence = gr.Textbox(label='Confidence')
            start_btn = gr.Button(value='Submit', elem_classes=["gr-button"])
    start_btn.click(fn=lunch, inputs=input_text, outputs=[class_name, confidence])

demo.launch()