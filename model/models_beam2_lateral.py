import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from utils.util import share_embedding
from utils.config import config
from utils.util import BeamSearchNode
from queue import PriorityQueue
import operator

class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size, num_classes):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # Frontal
        resnet_frontal = models.resnet18(pretrained=True)
        modules_frontal = list(resnet_frontal.children())[:-1]      # delete the last fc layer.
        self.resnet_frontal = torch.nn.Sequential(*modules_frontal)
        # Lateral
        resnet_lateral = models.resnet18(pretrained=True)
        modules_lateral = list(resnet_lateral.children())[:-1]      # delete the last fc layer.
        self.resnet_lateral = torch.nn.Sequential(*modules_lateral)
        
        self.dropout = torch.nn.Dropout(0.1)
        self.merge_linear = torch.nn.Linear(2*resnet_frontal.fc.in_features, 2*resnet_frontal.fc.in_features)
        self.feature_linear = torch.nn.Linear(2*resnet_frontal.fc.in_features, embed_size)
        self.classfy_linear = torch.nn.Linear(2*resnet_frontal.fc.in_features, num_classes)
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
#         self.norm2d = torch.nn.BatchNorm2d(3, eps=1e-06)
        
    def forward(self, images_frontal, images_lateral):
        """Extract feature vectors from input images."""
        # Frontal
        resnet_output_frontal = self.resnet_frontal(images_frontal)
        resnet_output_frontal = resnet_output_frontal.reshape(resnet_output_frontal.size(0), -1)
        resnet_output_frontal = self.dropout(resnet_output_frontal)
        # Lateral
        resnet_output_lateral = self.resnet_lateral(images_lateral)
        resnet_output_lateral = resnet_output_lateral.reshape(resnet_output_lateral.size(0), -1)
        resnet_output_lateral = self.dropout(resnet_output_lateral)
        # Merge
        resnet_output = torch.cat([resnet_output_frontal, resnet_output_lateral], dim=-1)
        resnet_output = self.merge_linear(resnet_output)
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
        self.lstm = torch.nn.LSTM(2*embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, vocab.n_words)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-06)
        self.max_seg_length = max_seq_length
    
    
    def forward(self, features, captions, lengths, prev_state):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = self.norm(embeddings)
        embeddings = torch.cat((features.unsqueeze(1).repeat(1,embeddings.size(1),1), embeddings), 2)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, state = self.lstm(packed, prev_state)
        rnn_out = self.dropout(hiddens[0])
        outputs = self.linear(rnn_out)
        return outputs, state
    
    
    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(config.device, non_blocking=True),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(config.device, non_blocking=True))

    
    def greedy_decode(self, features):
        states = None
        decode_outputs = []
        length = [1]
        ys = torch.ones(1,1).fill_(config.SOS_idx).long().to(config.device, non_blocking=True)
        while True:
            if ys.item() == config.EOS_idx:
                break
            output, states = self(features, ys, length, states)
            _, ys = torch.max(output,dim=1)
#             ys = torch.multinomial(torch.softmax(output,dim=-1), 1)
            decode_outputs.append(ys.item())
            ys = ys.unsqueeze(1)
        return decode_outputs
    
    def beam_decode(self, features, beam_width=10):
        """
        Code adapted from: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/9f6b66f43d2e05175dabcc024f79e1d37a667070/decode_beam.py
        Currently can only take a batch size of 1
        """
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        for idx in range(1):
            decoder_hidden = self.zero_state(batch_size=1)
#             decoder_hidden = None
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[config.SOS_idx]]).to(config.device)
            
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()
            # start the queue
            nodes.put( (-node.eval(), node) )
            qsize = 1
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 300: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == config.EOS_idx and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                decoder_output, decoder_hidden = self(features, decoder_input, [1], decoder_hidden)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


