import torch
import numpy as np

import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import transforms
import sys
from pycocotools.coco import COCO
from src.data_loader import get_loader
from .context import DecoderRNN, EncoderCNN
from .context import Vocabulary
import math
from .context import COCO_SMALL
from itertools import chain

def test_train():
    COCOAPI_LOC = COCO_SMALL

    ## TODO #1: Select appropriate values for the Python variables below.
    batch_size = 64  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 10  # number of training epochs
    save_every = 1  # determines frequency of saving model weights
    print_every = 100  # determines window for printing average loss
    log_file = 'training_log.txt'  # name of file with saved training loss and perplexity

    # (Optional) TODO #2: Amend the image transform below.
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Build data loader.
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=vocab_from_file,
                             cocoapi_loc=COCOAPI_LOC
                             )

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # TODO #3: Specify the learnable parameters of the model.
    params_1 = decoder.parameters()
    params_2 = encoder.parameters()
    both = chain(params_1, params_2)

    # TODO #4: Define the optimizer.
    lr_o_d = 0.001
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=lr_o_d)
    lr_o_e = 0.001
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr_o_e)
    optimizer = torch.optim.Adam(both, lr=lr_o_e)

    # Set the total number of training steps per epoch.
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
