import os
import numpy as np
from PIL import Image
import torch
from config import config

class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, reports, transform=None, return_finding=False,
                 images_dir=config.image_dir, file_list=config.file_list):
        self.transform = transform
        self.return_finding = return_finding
        if images_dir[-1] != "/": images_dir = images_dir + "/"
        file_lines = [line.rstrip("\n") for line in open(file_list)][1:]
        self.reports = reports
        self.frontal_images = []
        self.lateral_images = []
        self.problems = []
        self.findings = []
        self.impressions = []
        self.vocab = []
        self.classes = get_categories()
        self.num_classes = len(self.classes)
        # build uid -> image mapping
        self.uid_to_images = {}
        for line in file_lines:
            line = line.split(",")
            image_path = images_dir+line[1]
            uid = str(int(line[0]))
            if os.path.isfile(image_path):
                try: self.uid_to_images[uid]
                except: self.uid_to_images[uid] = [None, None]
                if line[-1] == "Frontal":
                    self.uid_to_images[uid][0] = image_path
                elif line[-1] == "Lateral":
                    self.uid_to_images[uid][1] = image_path
        # build image -> report mapping
        for uid in list(self.reports.keys()):
            frontal_img_path, lateral_img_path = self.uid_to_images[uid]
            if frontal_img_path and lateral_img_path:
                problem, finding, impression = self.reports[uid]
                finding, impression = finding.strip().lower(), impression.strip().lower()
                new_vocab = list(set(finding+impression))
                self.frontal_images.append(frontal_img_path)
                self.lateral_images.append(lateral_img_path)
                self.problems.append(problem)
                self.findings.append(finding)
                self.impressions.append(impression)
                self.vocab += new_vocab
        
        self.vocab = list(set(self.vocab))
        self.tokenizer = Tokenizer(char_set=self.vocab)
        
    def __len__(self):
        return len(self.frontal_images)

    def __getitem__(self, index):
        image_path, finding, impression = self.frontal_images[index], self.findings[index], self.impressions[index]
        #image = Image.open(image_path).convert('L')
        image = Image.open(image_path).convert('RGB')
        class_label = self.problems[index]
        one_hot = [0 for _ in range(self.num_classes)]
        for p in class_label:
            one_hot[self.classes.index(p)] += 1
        class_label = torch.from_numpy(np.asarray(one_hot, dtype="float"))
        if self.transform:
            image = self.transform(image)
        impression = torch.from_numpy(self.tokenizer.encode(impression))
        if self.return_finding:
            finding = torch.from_numpy(self.tokenizer.encode(finding))
            return image, class_label, finding, impression
        else:
            return image, class_label, impression
    
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, class_labels, captions = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    class_labels = torch.stack(class_labels, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, class_labels, targets, lengths

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
    
    
def get_categories():
    return [
        'airspace disease', 'aorta', 'aorta, thoracic', 'arthritis', 'atherosclerosis', 'bone diseases, metabolic', 'calcified granuloma', 'calcinosis', 'cardiac shadow', 'cardiomegaly', 'catheters, indwelling', 'cicatrix', 'consolidation', 'costophrenic angle', 'deformity', 'density', 'diaphragm', 'diaphragmatic eventration', 'emphysema', 'fractures, bone', 'granuloma', 'granulomatous disease', 'hernia, hiatal', 'implanted medical device', 'infiltrate', 'kyphosis', 'lung', 'markings', 'no indexing', 'nodule', 'normal', 'opacity', 'osteophyte', 'other', 'pleural effusion', 'pneumonia', 'pulmonary atelectasis', 'pulmonary congestion', 'pulmonary disease, chronic obstructive', 'pulmonary edema', 'pulmonary emphysema', 'scoliosis', 'spine', 'spondylosis', 'surgical instruments', 'technical quality of image unsatisfactory ', 'thickening', 'thoracic vertebrae'
    ]