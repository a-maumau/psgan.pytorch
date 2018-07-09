import torchtext.vocab as vocab
import torchvision.transforms as transforms

from data_loader import DTDDataLoader, ImageDataLoader

# it is best to pre-calculate mean and std, or normalize at batch.

def get_dtd_train_loader(args, img_size):
    transform = transforms.Compose([#transforms.Resize(img_size),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

    dataset_args = [
                    args.dataset,
                    args.image_list,
                    transform
                ]

    return DTDDataLoader(*dataset_args)

def get_train_loader(args, img_size):
    transform = transforms.Compose([#transforms.Resize(img_size),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

    dataset_args = [
                    args.dataset,
                    transform
                ]

    return ImageDataLoader(*dataset_args)
