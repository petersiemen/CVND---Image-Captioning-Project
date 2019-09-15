from .context import CoCoDataset
import os
from torchvision import transforms
import torch.utils.data as data

def test_coco_dataset():
    transform_train = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    mode = "train"
    batch_size=5
    vocab_threshold = 5
    vocab_file='../vocab.pkl'
    start_word="<start>"
    end_word="<end>"
    unk_word="<unk>"
    vocab_from_file=True
    cocoapi_loc='/opt'
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
    annotations_file = '/home/peter/datasets/coco/annotations/annotations/captions_train2014.json'
    num_workers = 4

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
                                  num_workers=num_workers
                                  )

    first_items = dict(list(data_loader.dataset.vocab.word2idx.items())[:10])
    print(first_items)

    print(len(data_loader))

