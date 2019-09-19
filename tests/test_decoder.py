import torch
import numpy as np
from .context import DecoderRNN
import torch.nn as nn


def test_decoder():
    vocab_size = 24
    embed_size = 256
    hidden_size = 512
    batch_size = 10
    sequence_length = 5

    features = np.random.rand(batch_size, embed_size)
    features = torch.from_numpy(features).type(torch.FloatTensor)
    print(features.shape)

    captions = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    captions = torch.from_numpy(captions)
    print(captions.shape)

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, 2)
    loss_function = nn.NLLLoss()
    print(decoder)

    outputs = decoder(features, captions)
    print(outputs)
    print(outputs.shape)
    #loss = loss_function(outputs, captions)
    reshaped_output = outputs.view(-1, vocab_size, batch_size)
    reshaped_captions = captions.view(-1, batch_size)
    loss = loss_function(reshaped_output, reshaped_captions)
    print(loss)
    loss.backward()


    # unqueeezed = features.unsqueeze(1)
    # print(features.shape)
    # print(unqueeezed.shape)

    #
    # outputs = decoder(features, captions)
    # print(outputs.shape)
    #
    # assert outputs.view(-1, vocab_size).shape == captions.view(-1).shape
