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
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        losses.append(loss.item())

        if i == 0:
            image = images[0]
            caption = captions[0]
            output = outputs[0]
            sentence = clean_sentence([i.item() for i in caption], data_loader)
            output_idx = torch.argmax(output, dim=1)
            prediction = clean_sentence([i.item() for i in output_idx], data_loader)

    return np.mean(losses), sentence, prediction, image
    # word_idx_output = torch.argmax(outputs, dim=2)

    # scores = []
    # for i in range(len(captions)):
    #    reference = clean_sentence(captions[i].numpy(), data_loader)
    #    candidate = clean_sentence(word_idx_output[i].numpy(), data_loader)
    #    score = sentence_bleu([word_tokenize(reference)], word_tokenize(candidate))
    #    scores.append(score)

    # return sum(scores) / len(scores), loss
    # return loss
