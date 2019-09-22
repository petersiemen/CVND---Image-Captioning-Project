from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import torch
from .clean_sentence import clean_sentence
import numpy as np


def validate(encoder, decoder, data_loader, criterion, vocab_size):
    # set the networks into validation mode
    encoder.eval()
    decoder.eval()
    losses = []
    for i, (images, captions) in enumerate(iter(data_loader)):
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        losses.append(loss.item())
    return np.mean(losses)
    # word_idx_output = torch.argmax(outputs, dim=2)

    # scores = []
    # for i in range(len(captions)):
    #    reference = clean_sentence(captions[i].numpy(), data_loader)
    #    candidate = clean_sentence(word_idx_output[i].numpy(), data_loader)
    #    score = sentence_bleu([word_tokenize(reference)], word_tokenize(candidate))
    #    scores.append(score)

    # return sum(scores) / len(scores), loss
    # return loss
