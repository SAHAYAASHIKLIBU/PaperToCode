from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

class Dataset:
    def __init__(self, config):
        self.config = config

        self._build_dataset()
        self._build_dataloader()

    def _build_transform(self):
        transforms = [
            Resize((self.config.img_size, self.config.img_size)),
            ToTensor(),
        ]

        # CIFAR-10 mean & std
        if self.config.normalize:
            transforms.append(
                Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                )
            )

        return Compose(transforms)

    def _build_dataset(self):
        transform = self._build_transform()

        self.train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        self.test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
        self.test_infer = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
        )
        self.classes = self.train_dataset.classes
        self.idToclasses = {v: k for k, v in self.train_dataset.class_to_idx.items()}

    def _build_dataloader(self):

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.train_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.test_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def __len__(self):
        return len(self.train_dataset)

    def num_classes(self):
        return len(self.classes)