from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import torch
from .clean_sentence import clean_sentence
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(encoder, decoder, data_loader, criterion, vocab_size):
    # set the networks into validation mode
    encoder.eval()
    decoder.eval()
    losses = []
    sentence = ''
    prediction = ''
    image = ''
    for i, (images, captions) in enumerate(iter(data_loader)):
        images = images.to(device)
        captions = captions.to(device)

        features = encoder(images)
        word_idxs, fc_out = decoder.sample(features, max_len=captions.size(1))
        loss = criterion(fc_out.view(-1, vocab_size), captions.view(-1))
        losses.append(loss.data)
    return np.mean(losses)

    # scores = []
    # for i in range(len(captions)):
    #    reference = clean_sentence(captions[i].numpy(), data_loader)
    #    candidate = clean_sentence(word_idx_output[i].numpy(), data_loader)
    #    score = sentence_bleu([word_tokenize(reference)], word_tokenize(candidate))
    #    scores.append(score)

    # return sum(scores) / len(scores), loss
    # return loss
