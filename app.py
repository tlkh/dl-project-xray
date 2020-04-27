import streamlit as st
import ast
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import csv
import dataset_word as data
import models_attn as models
import nltk
from nltk.translate.bleu_score import sentence_bleu

class Config:
    cleaned_reports = "./xray-dataset/cleaned_reports.csv"
    image_dir = "./xray-dataset/images/images_normalized/"
    file_list = "./xray-dataset/indiana_projections.csv"
    pretrained_emb = True    
    emb_file = "./vectors/glove.6B.300d.txt"
    PAD_idx = 0
    UNK_idx = 1
    EOS_idx = 2
    SOS_idx = 3
    emb_dim = 300
    hidden_dim = 512
    num_layers = 1
    batch_size = 1
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()

def main():
    st.sidebar.title("X-Ray Image Captioning")
    st.sidebar.text("50.039 Deep Learning Project")
    run_the_app()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    train_dataset, valid_dataset, test_dataset, num_classes, tokenizer = load_dataset()
    st.sidebar.subheader("\nControls")
    option = st.sidebar.selectbox("Choose between training, validation and test set:",
                          ("Train", "Validation", "Test"))
    if option == "Train":
        dataset = train_dataset
    elif option == "Validation":
        dataset = valid_dataset
    else:
        dataset = test_dataset
    selected_index = st.sidebar.slider("Pick an X-ray image from the " + option + " dataset", 0, len(dataset)-1, 0)
    image, _, impression = dataset.__getitem__(selected_index)
    image_numpy = np.moveaxis(image.numpy(), 0, -1)
    show_attention = st.sidebar.checkbox("Show Attention Map")
    preds, alpha = infer(image, num_classes, tokenizer)
    if show_attention:
        image_numpy = image_numpy*0.2 + alpha*0.8
    image_numpy = np.clip(image_numpy, 0.0, 1.0)
    st.image(image_numpy, caption=option+" X-Ray image "+str(selected_index), width=480)
    original = tokenizer.decode(impression[1:-1])
    generated = tokenizer.decode(preds).split("EOS")[0]
    reference = [nltk.word_tokenize(original)]
    candidate = nltk.word_tokenize(generated)
    bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    st.write("**Original**:\n", original)
    st.write("**Generated**:\n", generated)
    st.write("**BLEU score**:\n", str(bleu_score))

@st.cache
def load_dataset():
    def parse_list(input_str):    
        return ast.literal_eval(input_str)
    reports = {}
    with open(config.cleaned_reports) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                uid, problems, findings, impression = row[1:]
                reports[str(uid)] = (parse_list(problems), findings, impression)
    def create_report_splits(reports, seed=1337):
        uid_list = list(reports.keys())
        train_uids, valtest_uids = train_test_split(uid_list, test_size=0.2, random_state=seed)
        valid_uids, test_uids = train_test_split(valtest_uids, test_size=0.5, random_state=seed)
        train_reports = {}
        valid_reports = {}
        test_reports = {}
        splits = [train_uids, valid_uids, test_uids]
        output_reports = [train_reports, valid_reports, test_reports]

        for i in range(len(splits)):
            for uid in splits[i]:
                output_reports[i][str(uid)] = reports[str(uid)]

        return output_reports
    train_reports, valid_reports, test_reports = create_report_splits(reports)
    train_dataset = data.XRayDataset(
        reports=train_reports,
        transform=transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ]
    ))
    valid_dataset = data.XRayDataset(
        reports=valid_reports,
        transform=transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ]
    ))
    test_dataset = data.XRayDataset(
        reports=test_reports,
        transform=transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ]
    ))
    return train_dataset, valid_dataset, test_dataset, len(train_dataset.classes), train_dataset.tokenizer
    

def infer(image, num_classes, tokenizer):
    @st.cache(allow_output_mutation=True)
    def load_network(num_classes, tokenizer):
        encoder = models.EncoderCNN(num_classes).to(config.device)
        decoder = models.AttnDecoderRNN(attention_dim=config.hidden_dim,
                                embed_dim=config.emb_dim,
                                decoder_dim=config.hidden_dim,
                                vocab_size=tokenizer.n_words,
                                encoder_dim=2048,
                                device=config.device).to(config.device)
        encoder.load_state_dict(torch.load("./saved_exp/revised_model/encoder_word.pt"))
        decoder.load_state_dict(torch.load("./saved_exp/revised_model/decoder_word.pt"))
        return encoder, decoder
    encoder, decoder = load_network(num_classes, tokenizer)
    
    image_tensor = image.unsqueeze(0).to(config.device)
    logits, features = encoder(image_tensor)
    
    seed = []
    seed = torch.from_numpy(tokenizer.encode(seed)).unsqueeze(0).cuda()
    predictions, seed, decode_lengths, alphas = decoder.sample(features, seed, [32, ])
    alphas = alphas.reshape([1, 31, 10, 10])
    alphas = torch.sum(alphas, 1).squeeze(0)
    alphas = cv2.resize(alphas.detach().cpu().numpy(), (299,299))
    alphas = np.clip(alphas, 0.0, 1.0)
    alphas = cv2.cvtColor(alphas, cv2.COLOR_GRAY2RGB)
    alphas[:,:,1] = 0
    sampled_ids = list(predictions[0].cpu().numpy())
    return sampled_ids, alphas

if __name__ == "__main__":
    main()