import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--outdir", default="./saved_exp/wordLevel_model2/")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()


import os;
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import apex
import csv
import utils.dataset_word as data
import model.models_beam2 as models
import ast
from tqdm import trange
from sklearn.model_selection import train_test_split
# from tqdm.notebook import tqdm
from fastprogress.fastprogress import master_bar, progress_bar
# from config import config
import nltk
from nltk.translate.bleu_score import sentence_bleu
from utils.evaluate import *



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
    batch_size = args.batch_size
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()


num_epochs = args.epochs
pretrained = args.pretrained
memory_format = torch.channels_last
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

def parse_list(input_str):    
    return ast.literal_eval(input_str)

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

def main():
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

    train_reports, valid_reports, _ = create_report_splits(reports)

    train_dataset = data.XRayDataset(
        reports=train_reports,
        transform=transforms.Compose([
            transforms.Resize(299),
            transforms.RandomCrop((299,299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    ))
    train_dataloader = torch.utils.data.dataloader.DataLoader(train_dataset,
                                                              collate_fn=data.collate_fn,
                                                              pin_memory=True,
                                                              shuffle=True,
                                                              drop_last=True,
                                                              batch_size=config.batch_size,
                                                              num_workers=config.batch_size)
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
    valid_dataset.tokenizer = train_dataset.tokenizer
    valid_dataloader = torch.utils.data.dataloader.DataLoader(valid_dataset,
                                                              collate_fn=data.collate_fn,
                                                              pin_memory=True,
                                                              shuffle=True,
                                                              drop_last=True,
                                                              batch_size=config.batch_size,
                                                              num_workers=config.batch_size)

    num_classes = len(train_dataset.classes)

    encoder = models.EncoderCNN(config.emb_dim, num_classes).to(config.device, memory_format=memory_format)
    decoder = models.DecoderRNN_Word(config.emb_dim, config.hidden_dim, train_dataset.tokenizer, config.num_layers).to(config.device, memory_format=memory_format)

    classes_loss = torch.nn.BCEWithLogitsLoss()
    outputs_loss = torch.nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = apex.optimizers.FusedAdam(params, lr=config.learning_rate)

    [encoder, decoder], optimizer = apex.amp.initialize([encoder, decoder], optimizer, opt_level="O1")

    def train_one_epoch(dataloader, batch_size, encoder, decoder, classes_loss, outputs_loss, optimizer, train=True):
        total_step = len(dataloader.dataset)//batch_size
        if train:
            encoder.train()
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()
        running_c_loss = torch.Tensor([0.0])
        running_o_loss = torch.Tensor([0.0])
        state_h, state_c = decoder.zero_state(batch_size)
        with torch.set_grad_enabled(train):
            for i, (images, class_labels, captions, lengths) in enumerate(progress_bar(dataloader)):
                images = images.to(config.device, non_blocking=True).contiguous(memory_format=memory_format)
                captions = captions.to(config.device, non_blocking=True)
                class_labels = class_labels.to(config.device, non_blocking=True)
                lengths = [o-1 for o in lengths]
                targets = torch.nn.utils.rnn.pack_padded_sequence(captions[:,1:], lengths, batch_first=True, enforce_sorted=False)[0]
                encoder.zero_grad()
                decoder.zero_grad()
                logits, features = encoder(images)
                c_loss = classes_loss(logits, class_labels)
                outputs, (state_h, state_c) = decoder(features, captions[:,:-1], lengths, (state_h, state_c))
                o_loss = outputs_loss(outputs, targets)
                if train:
                    with apex.amp.scale_loss(c_loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                    with apex.amp.scale_loss(o_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    state_h = state_h.detach()
                    state_c = state_c.detach()
                    optimizer.step()
                running_c_loss += c_loss
                running_o_loss += o_loss
        c_loss = float(running_c_loss.item()/total_step)
        o_loss = float(running_o_loss.item()/total_step)
        return c_loss, o_loss

    batch_size = config.batch_size
    
    if not args.test:
        print("Start training")
    
        history = {
            "train_c_loss": [],
            "train_o_loss": [],
            "valid_c_loss": [],
            "valid_o_loss": []
        }

        best_loss = 100

        for epoch in range(num_epochs):
            print("\nEpoch", epoch+1, "/", num_epochs, ":\n")

            train_c_loss, train_o_loss = train_one_epoch(train_dataloader, batch_size, encoder, decoder, classes_loss, outputs_loss, optimizer, train=True)
            print("* train_loss - ", round(train_c_loss,3),round(train_o_loss,3), "- perplexity -", round(np.exp(train_o_loss),3))
            history["train_c_loss"].append(train_c_loss)
            history["train_o_loss"].append(train_o_loss)

            valid_c_loss, valid_o_loss = train_one_epoch(valid_dataloader, batch_size, encoder, decoder, classes_loss, outputs_loss, optimizer, train=False)
            print("* valid_loss - ", round(valid_c_loss,3),round(valid_o_loss,3), "- perplexity -", round(np.exp(valid_o_loss),3))
            history["valid_c_loss"].append(valid_c_loss)
            history["valid_o_loss"].append(valid_o_loss)

            current_valid_loss = valid_o_loss
            if current_valid_loss < best_loss:
                print("* best loss, saving weights")
                best_loss = current_valid_loss
                torch.save(encoder.state_dict(), outdir+"encoder_word.pt")
                torch.save(decoder.state_dict(), outdir+"decoder_word.pt")

        print("Save history to CSV file")
        df = pd.DataFrame(list(zip(history["train_c_loss"],
                                   history["train_o_loss"],
                                   history["valid_c_loss"],
                                   history["valid_o_loss"])),
                          columns =["train_c_loss",
                                    "train_o_loss",
                                    "valid_c_loss",
                                    "valid_o_loss"])
        df.to_csv(outdir+"history.csv")

    print("Load weights and run mAP and BLEU eval")
    
    encoder.load_state_dict(torch.load(outdir+"encoder_word.pt"))
    decoder.load_state_dict(torch.load(outdir+"decoder_word.pt"))
    y_true, y_pred = get_class_predictions(encoder, train_dataset)
    recall, precision, AP, train_mAP = evaluate_encoder_predictions(y_true, y_pred)

    y_true, y_pred = get_class_predictions(encoder, valid_dataset)
    recall, precision, AP, valid_mAP = evaluate_encoder_predictions(y_true, y_pred)

    print("* train mAP -", round(train_mAP, 3), "- valid mAP -", round(valid_mAP, 3))
    
    bleu_scores = []

    for name, dataloader in zip(["train", "valid"],[train_dataloader,valid_dataloader]):
        encoder.eval()
        decoder.eval()
        running_bleu = 0.0
        dataset_len = len(dataloader.dataset)
        with torch.set_grad_enabled(False):
            for index in trange(0, dataset_len):
                image, problems, impression = dataloader.dataset.__getitem__(index)
                image_tensor = image.unsqueeze(0).to(device)
                logits, features = encoder(image_tensor)
#                 seed = []
#                 seed = torch.from_numpy(train_dataset.tokenizer.encode(seed)).unsqueeze(0).cuda()
#                 predictions, seed, decode_lengths, alphas = decoder.sample(features, seed, [32, ])
#                 sampled_ids = list(predictions[0].cpu().numpy())
                # sampled_ids = decoder.beam_decode(features)
                sampled_ids = decoder.greedy_decode(features)
                sampled_ids = [i for i in sampled_ids]
                original = train_dataset.tokenizer.decode(impression[1:-1])
                generated = train_dataset.tokenizer.decode(sampled_ids[:-1])
                reference = [nltk.word_tokenize(original)]
                candidate = nltk.word_tokenize(generated)
                bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                running_bleu += bleu_score
            bleu_score = running_bleu/dataset_len
            bleu_scores.append(bleu_score)
            
    print("* train/valid BLEU-1 scores", bleu_scores)

if __name__ == "__main__":
    main()
