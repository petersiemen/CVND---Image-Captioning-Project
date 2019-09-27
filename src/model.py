import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    def get_learnable_parameters(self, ):
        return [param for name, param in self.named_parameters() if name.startswith('embed')]


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # define model layers
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, features, captions):
        # batch size
        batch_size = features.size(0)

        hidden_state, cell_state = self.init_hidden(batch_size, features)

        # define the output tensor placeholder
        outputs = torch.zeros((batch_size, captions.size(1), self.vocab_size)).to(device)

        # embed the captions
        captions_embed = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1) - 1):
            out, (hidden_state, cell_state) = self.lstm(captions_embed[:, t, :].view(batch_size, 1, -1),
                                                        (hidden_state, cell_state))
            out = out.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            # build the output tensor
            outputs[:, t + 1, :] = out

        return outputs

    def sample(self, inputs, states=None, max_len=20, top_k=5):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        batch_size = inputs.size(0)
        # list of word indices
        word_idxs = torch.zeros((batch_size, max_len)).type(torch.LongTensor)
        outputs = torch.zeros((batch_size, max_len, self.vocab_size)).to(device)

        def sample_word_from_fc_out(fc_out):
            p = F.softmax(fc_out, dim=1).data
            p, top_ch = p.topk(top_k)
            word_idxs_for_batch = torch.zeros(batch_size, dtype=torch.long)
            for i in range(top_ch.size(0)):
                top_ch_i = top_ch[i].cpu().numpy().squeeze()
                # select the likely next character with some element of randomness
                p_i = p[i].cpu().numpy().squeeze()
                word_idx = np.random.choice(top_ch_i, p=p_i / p_i.sum())
                word_idxs_for_batch[i] = int(word_idx)
            return word_idxs_for_batch

        hidden_state, cell_state = self.init_hidden(batch_size, inputs)

        word_idx_for_batch = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
        for i in range(max_len - 1):
            embedding = self.embed(word_idx_for_batch.to(device))
            lstm_out, (hidden_state, cell_state) = self.lstm(embedding.view(batch_size, 1, -1),
                                                             (hidden_state, cell_state))
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
            fc_out = self.fc(lstm_out)
            outputs[:, i, :] = fc_out
            word_idx_for_batch = sample_word_from_fc_out(fc_out)
            word_idxs[:, i] = word_idx_for_batch

        return word_idxs, outputs

    def init_hidden(self, batch_size, features):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function

        # initialize hidden state with zero weights, and move to GPU if available

        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden_state, cell_state = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                                        weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden_state, cell_state = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                                        weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        # initialized the first hidden layer with the extracted features from the CNN
        hidden_state[0, :, :] = features.view(1, batch_size, self.hidden_size)
        cell_state[0, :, :] = features.view(1, batch_size, self.hidden_size)
        return hidden_state, cell_state
