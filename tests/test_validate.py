from src.data_loader import get_val_loader
from torchvision import transforms
from .context import COCO_SMALL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from context import clean_sentence

HERE = os.path.dirname(os.path.abspath(__file__))


def test_validate():
    batch_size = 5  # batch size
    vocab_threshold = 5  # minimum word count threshold

    img_folder = COCO_SMALL + '/cocoapi/images/val2014/'
    annotations_file = COCO_SMALL + '/cocoapi/annotations/captions_val2014.json'

    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    val_data_loader = get_val_loader(transform=transform_train,
                                     batch_size=batch_size,
                                     vocab_threshold=vocab_threshold,
                                     annotations_file=annotations_file,
                                     img_folder=img_folder
                                     )

    images, captions = next(iter(val_data_loader))

    transform_train = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
    ])

    for i in range(len(images)):
        image = images[i]
        caption = captions[i]
        pil_image = transform_train(image)
        numpy_image = np.transpose(pil_image.numpy(), (1, 2, 0))
        plt.imshow(numpy_image)
        plt.show()
        sentence = clean_sentence(caption, val_data_loader)
        print(sentence)
