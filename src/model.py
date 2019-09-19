import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # define model layers
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, features, captions):
        # batch size
        batch_size = features.size(0)

        # TODO to cuda
        hidden_state, cell_state = self.init_hidden(batch_size)

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size))

        # embed the captions
        captions_embed = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                # features_with_mini_batch_dimension = features.view(batch_size, 1, -1)
                out, (hidden_state, cell_state) = self.lstm(features.view(batch_size, 1, -1),
                                                            (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                out, (hidden_state, cell_state) = self.lstm(captions_embed[:, t, :].view(batch_size, 1, -1),
                                                            (hidden_state, cell_state))

            out = out.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            # tag_scores = F.log_softmax(out, dim=1)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # list of word indices
        outputs = []

        hidden_state, cell_state = self.init_hidden(1)
        next = torch.zeros(1, 1, self.embed_size)
        for i in range(max_len):
            if i == 0:
                next, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))
            else:
                next, (hidden_state, cell_state) = self.lstm(next, (hidden_state, cell_state))
            out = next.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            word_idx = torch.argmax(out)
            next = self.embed(torch.tensor([[word_idx]]))
            outputs.append(word_idx.item())

        return outputs

    def init_hidden(self, batch_size):
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
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden
