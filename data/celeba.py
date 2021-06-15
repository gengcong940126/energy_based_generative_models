import torch
from torchvision import transforms, datasets
from data import dataset
class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)
def inf_train_gen(batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data = datasets.ImageFolder('/home/congen/code/AGE-exp/datasets/celeba_all', transform=transform)
    # data = dataset.FolderWithImages(root='/home/congen/code/AGE-exp/datasets/celeba_all',
    #                                    input_transform=transforms.Compose([
    #                                        transforms.Resize((64, 64)),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize(
    #                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                    ]),
    #                                    target_transform=transforms.ToTensor()
    #                                    )
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=0
    )
    train_loader = dict(train=InfiniteDataLoader(loader))
    # while True:
    #     for img, labels in loader:
    #         yield img
    return train_loader['train']