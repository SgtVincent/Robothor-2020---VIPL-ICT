from torchsummary import summary
import sys
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
import tensorwatch as tw
# from models.model_io import ModelInput, ModelOptions, ModelOutput
from utils.flag_parser import parse_arguments
import torch.jit as jit
# embed basemodel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init
from models import MatchModel

if __name__ == '__main__':

    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = MatchModel(args).to(device)

    # model_input = ModelInput()
    state = torch.zeros(1,512,7,7).to(device) # [1,512,7,7]
    hidden = (
        torch.zeros(1, args.hidden_state_sz).to(device), # [1,512]
        torch.zeros(1, args.hidden_state_sz).to(device), # [1,512]
    )
    target_class_embedding = torch.zeros(300).to(device)  # [300]
    action_probs = torch.zeros(1, args.action_space).to(device)  # [1, #(ACTION_SPACE)]

    # model_opts = ModelOptions()
    # tw.draw_model(model,([1,512,7,7],
    #                [1,args.hidden_state_sz],
    #                [1,args.hidden_state_sz],
    #                [1, 300],
    #                [1, args.action_space]),
    #               'model.png')
    # with SummaryWriter("model_vis",comment="basemodel") as writer:
    #     writer.add_graph(model, (state, hidden[0], hidden[1],
    #                              target_class_embedding, action_probs), verbose=True)
    # summary(model,[(512,7,7),
    #                (args.hidden_state_sz),
    #                (args.hidden_state_sz),
    #                (300),
    #                (args.action_space)])
    print(model)
