import torch
from torch.utils.data import TensorDataset


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    words = tuple(set(text))
    int_to_vocab = dict(enumerate(words))
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    special_chars = dict()
    special_chars['.'] = "||Period||"
    special_chars[','] = "||Comma||"
    special_chars['"'] = "||Quotation_Mark||"
    special_chars[';'] = "||Semicolon||"
    special_chars['!'] = "||Exclamation_mark||"
    special_chars['?'] = "||Question_mark||"
    special_chars['('] = "||Left_Parentheses||"
    special_chars[')'] = "||Right_Parentheses||"
    special_chars['-'] = "||Dash||"
    special_chars['\n'] = "||Return||"

    return special_chars




def create_tensors(words, sequence_length):
    length = len(words) - sequence_length
    feature_tensors = []
    target_tensors = []
    for start in range(0, length):
        feature_tensors.append(words[start:sequence_length + start])
        target_tensors.append(words[sequence_length + start])

    return torch.tensor(feature_tensors, dtype=torch.long), torch.tensor(target_tensors, dtype=torch.long)


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function

    feature_tensors, target_tensors = create_tensors(words, sequence_length)

    data = TensorDataset(feature_tensors, target_tensors)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size)

    return data_loader
