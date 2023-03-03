import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.transforms.transforms import *
from PIL import Image
from torch.utils.data import DataLoader
from lib import ForeverDataIterator
from torch.utils.data import Dataset

'''
import torch
from PIL import Image
from torch.utils.data import Dataset
#from .wheel import *
'''
def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageListT(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList(datasets.VisionDataset):

    def __init__(self, root: str, num_class: int, data_list_file: str, filter_class: list,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = self.parse_data_file(data_list_file, filter_class)
        self.num_class = num_class
        # self.class_to_idx = {cls: idx
        #                      for idx, clss in enumerate(self.classes)
        #                      for cls in clss}
        self.loader = default_loader

    def __getitem__(self, index: int, ) -> Tuple[Any, int]:
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def parse_data_file(self, file_name: str, filter_class: list) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                if target in filter_class:
                    data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return self.num_class

class ImageList1(datasets.VisionDataset):

    def __init__(self, root: str, num_class: int, data_list_file: str, filter_class: list,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = self.parse_data_file(data_list_file, filter_class)
        self.num_class = num_class
        # self.class_to_idx = {cls: idx
        #                      for idx, clss in enumerate(self.classes)
        #                      for cls in clss}
        self.loader = default_loader

    def __getitem__(self, index: int, ) -> Tuple[Any, int]:
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def parse_data_file(self, file_name: str, filter_class: list) -> List[Tuple[str, int]]:
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                sp = line.split()
                if(len(sp) == 3):
                    path, target = sp[0]+' '+sp[1],sp[2]
                else:
                    path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = float(target)
                if target in filter_class:
                    data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return self.num_class


class Office31(ImageList):
    """Office31 Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/1f5646f39aeb4d7389b9/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/05640442cd904c39ad60/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/a069d889628d4b468c32/?dl=1"),
        ("webcam", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/4c4afebf51384cf1aa95/?dl=1"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt"
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, data_list_file, filter_class, **kwargs):
        super(Office31, self).__init__(root, len(filter_class), data_list_file, filter_class, **kwargs)


class Xray(ImageList1):

    def __init__(self, root: str, data_list_file, filter_class, **kwargs):
        super(Xray, self).__init__(root, len(filter_class), data_list_file, filter_class, **kwargs)


train_transform1 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2),
                #, interpolation=Image.BICUBIC,fill=(255, 255, 255)),#fillcolor=(255, 255, 255), resample=Image.BICUBIC,
    CenterCrop(224),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform2 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[0]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform3 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2),
                #, interpolation=Image.BICUBIC,fill=(255, 255, 255)),
    FiveCrop(224),
    Lambda(lambda crops: crops[1]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform4 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                #, interpolation=Image.BICUBIC,fill=(255, 255, 255)),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[2]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform5 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[3]),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])


def esem_dataloader(args, filter_class):
    train_source_dataset1 = ImageList(root=args.root, num_class=len(filter_class), data_list_file=args.source,
                                      filter_class=filter_class, transform=train_transform1)
    esem_loader1 = DataLoader(train_source_dataset1, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    train_source_dataset2 = ImageList(root=args.root, num_class=len(filter_class), data_list_file=args.source,
                                      filter_class=filter_class, transform=train_transform2)
    esem_loader2 = DataLoader(train_source_dataset2, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    train_source_dataset3 = ImageList(root=args.root, num_class=len(filter_class), data_list_file=args.source,
                                      filter_class=filter_class, transform=train_transform3)
    esem_loader3 = DataLoader(train_source_dataset3, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    train_source_dataset4 = ImageList(root=args.root, num_class=len(filter_class), data_list_file=args.source,
                                      filter_class=filter_class, transform=train_transform4)
    esem_loader4 = DataLoader(train_source_dataset4, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    train_source_dataset5 = ImageList(root=args.root, num_class=len(filter_class), data_list_file=args.source,
                                      filter_class=filter_class, transform=train_transform5)
    esem_loader5 = DataLoader(train_source_dataset5, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)

    esem_iter1 = ForeverDataIterator(esem_loader1)
    esem_iter2 = ForeverDataIterator(esem_loader2)
    esem_iter3 = ForeverDataIterator(esem_loader3)
    esem_iter4 = ForeverDataIterator(esem_loader4)
    esem_iter5 = ForeverDataIterator(esem_loader5)

    return esem_iter1, esem_iter2, esem_iter3, esem_iter4, esem_iter5


'''
class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1
'''
