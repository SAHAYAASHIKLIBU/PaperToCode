from ntpath import isfile
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
import numpy as np
from torchvision import datasets
import random
import glob
import os

import wandb



class classify:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.transforms = Compose([ToTensor(),
                            Resize((self.config.img_size, self.config.img_size)),
                            Normalize(mean=(0.4914, 0.4822, 0.4465),\
                                 std=(0.2470, 0.2435, 0.2616))])
        self.resize = Resize((self.config.img_size, self.config.img_size))

    def predict(self, idToClass: dict, images: list | str | None = None):
        if images == None:
            images = self.pic_images_from_testset()
            images = glob.glob(self.config.dataset_images_path + "/*.jpg")
        if isinstance(images, list):
            fig, plots = plt.subplots(2, len(images)//2, figsize = (10, 5))
            plots = plots.flatten()
            if isinstance(images[0], str):
                for i in range(len(images)):
                    out = self._predict(images[i])
                    plots[i].imshow(Image.open(images[i]).convert("RGB"))
                    plots[i].set_title(f"Class : {idToClass[out]}")
                    plots[i].axis("off")
                plt.tight_layout()
                plt.savefig(self.config.plot_file_name)
                plt.close()
        elif isinstance(images, str):
            out = self._predict(images)
            plt.imshow(Image.open(images).convert("RGB"))
            plt.axis("off")
            plt.title(f"Class : {idToClass[out]}")
            plt.savefig(self.config.plot_file_name)
        else:
            print("Currently not supported")

    
    def _predict(self, img):
        img = Image.open(img).convert("RGB")
        img = self.transforms(img).unsqueeze(0)
        out = self.model(img).argmax()
        return int(out)

    def pic_images_from_testset(self, loader, count = 10, clear = False):
        if clear and os.path.isdir(self.config.dataset_images_path):
            shutil.rmtree(self.config.dataset_images_path)
        test_set = loader.test_infer
        os.makedirs(self.config.dataset_images_path, exist_ok = True)
        total = len(test_set)
        for i in range(count):
            idx = random.randint(0, total-1)
            img , label = test_set[idx]
            img = self.resize(img)
            img.save(f"{self.config.dataset_images_path}/test_{i+1}.jpg")
        return
       