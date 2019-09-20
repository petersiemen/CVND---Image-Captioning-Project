# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

COCO_SMALL = '/home/peter/datasets/coco-small'

from datasets.coco_dataset import CoCoDataset
from utils.clean_sentence import clean_sentence
from model import DecoderRNN
from model import EncoderCNN
from datasets.vocabulary import Vocabulary
