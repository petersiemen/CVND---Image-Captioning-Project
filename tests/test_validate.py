from src.data_loader import get_val_loader
from torchvision import transforms
from .context import COCO_SMALL


def test_validate():
    batch_size = 64  # batch size
    vocab_threshold = 5  # minimum word count threshold

    img_folder = COCO_SMALL + '/cocoapi/images/train2014/'
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

    print(images)

