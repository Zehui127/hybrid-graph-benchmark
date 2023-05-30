import hg.hybrid_graph.factory as factory
import torch
import torch.nn.functional as F
import torch.nn as nn


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model

class Average_Ensemble(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super.__init__()
        self.model1 = factory[info.model](info)
        self.model2 = factory[info.model](info)
        # initialize the two models with the checkpoint weights
        self.model1 = plt_model_load(self.model1, info["checkpoint1"])
        self.model2 = plt_model_load(self.model2, info["checkpoint2"])
    def forward(self, data, *args, **kargs):
        x1 = self.model1(data, *args, **kargs)
        x2 = self.model2(data, *args, **kargs)
        return (x1 + x2) / 2
