import torch
import numpy as np
from .context import DecoderRNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        with_mini_batch_dimension = embeds.view(len(sentence), 1, -1)
        lstm_out, (h1, h2) = self.lstm(embeds)
        #lstm_out, (h1, h2) = self.lstm(with_mini_batch_dimension)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def test_lstm():
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 4
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # with torch.no_grad():
    #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
    #     tag_scores = model(inputs)
    #     print(tag_scores)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            sentence_in2 = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
            tag_scores = model(sentence_in2)
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

def test_lstm_elem():
    batch_size = 3
    sequence_length = 6
    hidden_size = 4
    vocab_size = 5
    output_size = sequence_length

    embedding_dim = 12

    word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)

    sentence_in = torch.randint(vocab_size, size = (batch_size, sequence_length))

    embeddings = word_embeddings(sentence_in)

    out, (h1, h2) = lstm(embeddings)


    out_out= fc(out)


    print(out)
    print(out.shape)

    print(h1)
    print(h2)


def test_embedding():
    vocab_size = 6
    embedding_dim = 4
    word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    sentence_in = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    # sentence_in = torch.tensor([0, 1, 2, 3, 4, 5])
    print(sentence_in)
    print(sentence_in.shape)
    embeds = word_embeddings(sentence_in)
    print(embeds)
    print(embeds.shape)
    viewed = embeds.view(2, 1, 6, -1)
    print(viewed)
    print(viewed.shape)

    #
    #
    # unsqeezed = sentence_in.unsqueeze(1)
    # print(unsqeezed.shape)
    # print(sentence_in)
    # print(unsqeezed)
    #
    # features = np.random.rand(2, embedding_dim)
    #
