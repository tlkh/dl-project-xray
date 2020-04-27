import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from utils.util import share_embedding
from utils.config import config


class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size, num_classes, pretrained=True):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = torch.nn.Sequential(*modules)
        self.dropout = torch.nn.Dropout(0.1)
        self.feature_linear = torch.nn.Linear(resnet.fc.in_features, embed_size)
        self.classfy_linear = torch.nn.Linear(resnet.fc.in_features, num_classes)
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        resnet_output = self.resnet(images)
        resnet_output = resnet_output.reshape(resnet_output.size(0), -1)
        resnet_output = self.dropout(resnet_output)
        features = self.norm(self.feature_linear(resnet_output))
        logits = self.classfy_linear(resnet_output)
        return logits, features


class DecoderRNN(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=64):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths, prev_state):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = self.norm(embeddings)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        hiddens, state = self.lstm(packed, prev_state)
        rnn_out = self.dropout(hiddens[0])
        outputs = self.linear(rnn_out)
        return outputs, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class DecoderRNN_Word(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers, max_seq_length=64):
        """Set the hyper-parameters and build the layers."""
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = share_embedding(vocab, config.pretrained_emb)
        self.lstm = torch.nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, vocab.n_words)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths, prev_state):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = self.norm(embeddings)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        hiddens, state = self.lstm(packed, prev_state)
        rnn_out = self.dropout(hiddens[0])
        outputs = self.linear(rnn_out)
        return outputs, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids