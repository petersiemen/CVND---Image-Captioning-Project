import torch
import numpy as np
from .context import DecoderRNN
from .context import Vocabulary
import torch.nn as nn


def test_decoder():
    vocab_size = 24
    embed_size = 256
    hidden_size = 512
    batch_size = 10
    sequence_length = 5
    lr = 0.001

    features = np.random.rand(batch_size, embed_size)
    features = torch.from_numpy(features).type(torch.FloatTensor)
    print(features.shape)

    captions = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    captions = torch.from_numpy(captions)
    print(captions.shape)

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, 2)
    criterion = nn.CrossEntropyLoss()
    params = decoder.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    print(decoder)

    outputs = decoder(features, captions)
    print(outputs)
    print(outputs.shape)

    loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

    print(loss)
    loss.backward()
    opt.step()
