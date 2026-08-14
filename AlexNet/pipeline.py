from sympy.logic import inference
from config import TrainConfig, DatasetConfig, InferenceConfig
from model import AlexNet
from dataset import Dataset
from inference import Classify
from train import Trainer


data_config = DatasetConfig()
train_config = TrainConfig()
inference_config = InferenceConfig()
data = Dataset(data_config)
model = AlexNet()
train = Trainer(TrainConfig)
train.train(model, data)
predict = Classify(model, inference_config, data)
predict.predict()