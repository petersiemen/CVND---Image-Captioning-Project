def clean_sentence(output, data_loader):
    start_word = data_loader.dataset.vocab.start_word
    end_word = data_loader.dataset.vocab.end_word
    unk_word = data_loader.dataset.vocab.unk_word

    words = []
    for i in range(len(output)):
        word_idx = output[i]
        word = data_loader.dataset.vocab.idx2word.get(word_idx)
        if word == end_word:
            break
        elif word != start_word and word != unk_word:
            words.append(word)
    return " ".join(words)
