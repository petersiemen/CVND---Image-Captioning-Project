from .context import CoCoDataset
import os
from torchvision import transforms
import torch.utils.data as data
from src.data_loader import get_loader
from context import COCO_SMALL
from context import clean_sentence


def test_coco_dataset():
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    mode = "train"
    batch_size = 3
    vocab_threshold = 5
    vocab_file = '../vocab.pkl'
    start_word = "<start>"
    end_word = "<end>"
    unk_word = "<unk>"
    vocab_from_file = False
    cocoapi_loc = COCO_SMALL
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    dataset = CoCoDataset(transform=transform_train,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    # data loader for COCO dataset.
    data_loader = data.DataLoader(dataset=dataset,
                                  num_workers=4
                                  )

    images, captions = next(iter(data_loader))
    print(images.shape)
    print(captions.shape)


def test_data_loader():
    # Define a transform to pre-process the training images.
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    vocab_threshold = 5

    # Specify the batch size.
    batch_size = 10

    # Obtain the data loader.
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=False,
                             cocoapi_loc=COCO_SMALL  # uncomment for running on local
                             )
    print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

    images, captions = next(iter(data_loader))

    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)
    print(captions)


    print(data_loader.dataset.vocab.idx2word)
    for caption in captions:
        sentence = clean_sentence(caption, data_loader)
        print(caption)
        print(sentence)