import sys
# sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from src.data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from .context import DecoderRNN
from .context import EncoderCNN
from context import clean_sentence



def test_override_parameters():

    encoder = EncoderCNN(512)

    for name, param in encoder.named_parameters():
        print(name, '\t\t', param.shape)

    for param in encoder.get_learnable_parameters():
        print(param)


