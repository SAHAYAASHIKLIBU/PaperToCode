from config import TrainConfig, DatasetConfig, InferenceConfig
from model import AlexNet
from dataset import Dataset
from inference import Classify
from train import Trainer
import torch
import os


data_config = DatasetConfig()
train_config = TrainConfig()
inference_config = InferenceConfig()
data = Dataset(data_config)
model = AlexNet()
train = Trainer(TrainConfig)
train.train(model, data)

state_dict = torch.load(os.path.join(train_config.checkpoint_path, "best.pt"))
predict = Classify(model, inference_config, data)
predict.predict()