import os
import numpy as np
from PIL import Image
import torch

class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, return_finding=False,
                 images_dir="./images/images_normalized/",
                 file_list="./indiana_projections.csv",
                 reports_list="./indiana_reports.csv"):
        self.transform = transform
        self.return_finding = return_finding
        if images_dir[-1] != "/": images_dir = images_dir + "/"
        file_lines = [line.rstrip("\n") for line in open(file_list)][1:]
        reports_lines = [line.rstrip("\n") for line in open(reports_list)][1:]
        reports = {}
        for line in reports_lines:
            line = line.split(",")
            uid = str(int(line[0]))
            finding, impression = line[-2].strip().lower(), line[-1].strip().lower()
            reports[uid] = (finding, impression)
        print("Number of reports:", len(reports_lines))
        self.images = []
        self.findings = []
        self.impressions = []
        self.vocab = []
        skip = 0
        for line in file_lines:
            line = line.split(",")
            if line[-1] == "Frontal":
                image_path = images_dir+line[1]
                uid = str(int(line[0]))
                if os.path.isfile(image_path):
                    report = reports[uid]
                    finding, impression = report[0], report[1]
                    if len(finding) > 1 and len(impression) > 1:
                        self.images.append(image_path)
                        self.vocab += list(set(finding + impression))
                        self.findings.append(finding)
                        self.impressions.append(impression)
                else:
                    skip += 1
            else:
                skip += 1
        self.vocab = list(set(self.vocab))
        self.tokenizer = Tokenizer(char_set=self.vocab)
        print("Skipped:", skip, "images")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, finding, impression = self.images[index], self.findings[index], self.impressions[index]
        #image = Image.open(image_path).convert('L')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        impression = torch.from_numpy(self.tokenizer.encode(impression))
        if self.return_finding:
            finding = torch.from_numpy(self.tokenizer.encode(finding))
            return image, finding, impression
        else:
            return image, impression
    
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

class Tokenizer(object):
    def __init__(self, char_set):
        char_set = list(set(char_set))
        char_set.sort()
        self.char_set = char_set
    def encode(self, text, seq_len=None):
        return np.asarray([self.char_set.index(t) for t in text], dtype="int")
    def decode(self, sequence):
        return "".join([self.char_set[i] for i in sequence]).strip()
    def get_vocab(self):
        return self.char_set
    
    
    