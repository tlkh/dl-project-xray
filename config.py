import torch


class Config:
    # data
    cleaned_reports = "./xray-dataset/cleaned_reports.csv"
    image_dir = "./xray-dataset/images/images_normalized/"
    file_list = "./xray-dataset/indiana_projections.csv"
    pretrained_emb = True    
    emb_file = "./vectors/glove.6B.300d.txt"
    PAD_idx = 0
    UNK_idx = 1
    EOS_idx = 2
    SOS_idx = 3

    # model
    emb_dim = 300
    hidden_dim = 128
    num_layers = 3

    # training
    batch_size = 5
    learning_rate = 0.001
    
    # Others
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
