import tensorflow as tf
import torch
import datetime
from tensorflow import summary
import  torch.nn.functional as F
import time
import random


def test_filewriter():


    train_log_dir = 'test_log_dir'
    train_summary_writer = summary.create_file_writer(train_log_dir)

    name = "loss" + str(random.randint(1,10))

    with train_summary_writer.as_default():
        #name, tensor, collections = None, family = None):
        for i in range(10):
            loss = F.l1_loss(torch.rand(1), torch.rand(1))
            tf.summary.scalar(name, loss.item(), step=i)

