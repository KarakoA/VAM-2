from utils.graph_viz import make_dot

import torch
import numpy as np
import matplotlib.pyplot as plt

from config.configs import Config
from datasets.datasets import DatasetType
from datasets.locator import DatasetLocator
from train.trainer import Trainer

def run():
    config = Config()
    config.is_train = False
    config.batch_size = 3
    config.num_glimpses = 5
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    locator = DatasetLocator(config)

    dloader = locator.data_loader(DatasetType.TEST)
    trainer = Trainer(config, dloader)

    imgs,lbls = iter(dloader).next()
    (loss, _, _, _, _) =  trainer.one_batch(imgs,lbls)
    #dict(trainer.model.named_parameters()
    #params =dict(trainer.model.named_parameters())
    params = {**{'inputs': imgs}, **dict(trainer.model.named_parameters())}
    dot = make_dot(loss,params)
    dot.render('./tmp/dot-graph.gv', view = True)

if __name__ == '__main__':
    run()