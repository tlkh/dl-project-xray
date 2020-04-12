import torch


class Config:
    # data
    cleaned_reports = "./xray-dataset/cleaned_reports.csv"
    image_dir = "./xray-dataset/images/images_normalized/"
    file_list = "./xray-dataset/indiana_projections.csv"

    batch_size = 5
    emb_dim = 128
    hidden_dim = 128
    num_layers = 3
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
