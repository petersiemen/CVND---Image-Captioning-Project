import sys
# sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from src.data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .context import DecoderRNN
from .context import EncoderCNN
from context import clean_sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_test = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


def test_sample():
    vocab_size = 24
    embed_size = 256
    hidden_size = 512

    # TODO #1: Define a transform to pre-process the testing images.
    transform_test = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # -#-#-# Do NOT modify the code below this line. #-#-#-#

    # Create the data loader.
    data_loader = get_loader(transform=transform_test,
                             batch_size=2,
                             mode='train',
                             cocoapi_loc='/home/peter/datasets/coco-small',
                             vocab_threshold=1,
                             vocab_from_file=False
                             )

    print(data_loader.dataset.vocab.idx2word)
    vocab_size = len(data_loader.dataset.vocab.idx2word)

    # Obtain sample image before and after pre-processing.
    images, captions = next(iter(data_loader))


    encoder = EncoderCNN(hidden_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Obtain the embedded image features.
    features = encoder(images)
    # Pass the embedded image features through the model to get a predicted caption.
    word_idxs, fc_out = decoder.sample(features, max_len=captions.size(1))
    print('example output:', word_idxs)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(fc_out.view(-1, vocab_size), captions.view(-1))

    # assert (type(word_idxs) == list), "Output needs to be a Python list"
    # assert all([type(x) == int for x in word_idxs]), "Output should be a list of integers."
    # assert all([x in data_loader.dataset.vocab.idx2word for x in
    #             word_idxs]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

    for word_idx in word_idxs:
        print(clean_sentence([i.item() for i in word_idx], data_loader))
