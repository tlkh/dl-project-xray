import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from utils import share_embedding
from config import config


class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size, num_classes):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.inception_v3(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)
        self.dropout = torch.nn.Dropout(0.1)
        self.feature_linear = torch.nn.Conv2d(512, embed_size, kernel_size=(1, 1))
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
        # classifier
        self.global_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classify_linear = torch.nn.Linear(512, num_classes)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        resnet_output = self.resnet(images)
        resnet_output = self.dropout(resnet_output)
        features = resnet_output.permute(0, 2, 3, 1)
        pool_output = self.global_pool(resnet_output)
        flat_output = pool_output.reshape(pool_output.size(0), -1)
        logits = self.classify_linear(flat_output)
        return logits, features
    
    
class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = torch.nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = torch.nn.Linear(decoder_dim, attention_dim)
        self.full_att = torch.nn.Linear(attention_dim, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class AttnDecoderRNN(torch.nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, device, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.decode_step = torch.nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = torch.nn.Linear(encoder_dim, decoder_dim)
        self.init_c = torch.nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = [c-1 for c in caption_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths ])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas
    
    def sample(self, encoder_out, seed, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        seed_embed = self.embedding(seed)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = [c-1 for c in caption_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths)).long().to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)
        
        t = 0
        batch_size_t = sum([l > t for l in decode_lengths ])
        attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
        gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
        attention_weighted_encoding = gate * attention_weighted_encoding
        seed_embed = seed_embed[:batch_size_t, t, :]
        h, c = self.decode_step(
            torch.cat([seed_embed, attention_weighted_encoding], dim=1),
            (h[:batch_size_t], c[:batch_size_t]))
        preds = self.fc(self.dropout(h))
        _, preds = preds.max(1)
        predictions[:batch_size_t, t] = preds
        alphas[:batch_size_t, t, :] = alpha
        
        for t in range(1, max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths ])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            prev_token = predictions[:batch_size_t, t-1]
            prev_embed = self.embedding(prev_token)
            h, c = self.decode_step(
                torch.cat([prev_embed, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            _, preds = preds.max(1)
            predictions[:batch_size_t, t] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, seed, decode_lengths, alphas