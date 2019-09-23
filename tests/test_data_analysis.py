from src import seinfeld_helper
import numpy as np
from src.seinfeld_load_data import  *
import torch.nn as nn

def test_seinfeld_preprocess():
    data_dir = './Seinfeld_Scripts.txt'
    text = seinfeld_helper.load_data(data_dir)

    view_line_range = (0, 10)
    print(view_line_range)

    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

    lines = text.split('\n')
    print('Number of lines: {}'.format(len(lines)))
    word_count_line = [len(line.split()) for line in lines]
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))

    print()
    print('The lines {} to {}:'.format(*view_line_range))
    print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

    seinfeld_helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)



def test_seinfeld_data():
    sequence_length = 1  # of words in a sequence
    # Batch Size
    batch_size = 5

    int_text, vocab_to_int, int_to_vocab, token_dict = seinfeld_helper.load_preprocess()
    print(vocab_to_int)

    train_loader = batch_data(int_text, sequence_length, batch_size)
    inputs, labels = next(iter(train_loader))
    print(inputs)
    print(labels)

    n_vocab = len(int_to_vocab)
    n_embed = 16
    n_layers = 1
    embedding_dim = 16
    hidden_dim = 25
    dropout = 0

    embed = nn.Embedding(n_vocab, n_embed)
    lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                    dropout=dropout, batch_first=True)
    #
    #
    out = embed(inputs)
    print(out.shape)
    out, (h0, c0) = lstm.forward(out)
    print(out.shape)
