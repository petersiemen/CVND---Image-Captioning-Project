import os
import torch.utils.data as data
from .datasets.coco_dataset import CoCoDataset


def val_get_loader(transform,
                   annotations_file,
                   img_folder,
                   vocab_threshold=None,
                   batch_size=1,
                   num_workers=0
                   ):
    vocab_file = './vocab.pkl'
    start_word = "<start>"
    end_word = "<end>"
    unk_word = "<unk>"
    vocab_from_file = True
    dataset = CoCoDataset(transform=transform,
                          mode='train',  # to receive image, caption    pairs instead or orig_image, image
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    indices = dataset.get_train_indices()
    # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)

    return data.DataLoader(dataset=dataset,
                           num_workers=num_workers,
                           batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                   batch_size=dataset.batch_size,
                                                                   drop_last=False))
