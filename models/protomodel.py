from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput

# relnet_kernel_size = 3
# lstm_input_feature_map_size = 5

match_block_kernel_size = 1
feature_map_size = 7
action_channels = 10
fusion_channels = 64 * 2 + action_channels

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class ProtoModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        # target_embedding_sz = args.target_dim
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(ProtoModel, self).__init__()

        # self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1) # Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv1 = nn.Sequential(
                        nn.Conv2d(resnet_embedding_sz,64,kernel_size=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.fusion_block = nn.Sequential(
                        nn.Conv2d(fusion_channels,64,kernel_size=match_block_kernel_size),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()) # [64, 5,5]/[64, 7, 7]

        self.embed_action = nn.Linear(action_space, action_channels) # Linear(in_features=7, out_features=10, bias=True)

        # pointwise_in_channels = 64 + 10
        #
        # self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1) # Conv2d(138, 64, kernel_size=(1, 1), stride=(1, 1))

        lstm_input_sz = feature_map_size * feature_map_size * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz) # LSTMCell(1600, 512)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1) #Linear(in_features=512, out_features=1, bias=True)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs) # Linear(in_features=6272, out_features=7, bias=True)

        self.apply(weights_init)
        # relu_gain = nn.init.calculate_gain("relu")
        # self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_probs, params):

        action_embedding_input = action_probs
        # MODIFY
        state = state.view(-1, 512, 7, 7)
        target = target.view(-1, 512, 7, 7)

        if params is None:
            state_embedding = self.conv1(state)
            target_embedding = self.conv1(target)
            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(-1, 10, 1, 1).repeat(1, 1, feature_map_size, feature_map_size)

            x = torch.cat((state_embedding, target_embedding, action_reshaped), dim=1)
            x = self.fusion_block(x)
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:
            raise Exception("model for savn not implemented yet")
            # glove_embedding = F.relu(
            #     F.linear(
            #         target,
            #         weight=params["embed_glove.weight"],
            #         bias=params["embed_glove.bias"],
            #     )
            # )
            #
            # glove_reshaped = glove_embedding.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)
            #
            # action_embedding = F.relu(
            #     F.linear(
            #         action_embedding_input,
            #         weight=params["embed_action.weight"],
            #         bias=params["embed_action.bias"],
            #     )
            # )
            # action_reshaped = action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)
            #
            # image_embedding = F.relu(
            #     F.conv2d(
            #         state, weight=params["conv1.weight"], bias=params["conv1.bias"]
            #     )
            # )
            # x = self.dropout(image_embedding)
            # x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            #
            # x = F.relu(
            #     F.conv2d(
            #         x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
            #     )
            # )
            # x = self.dropout(x)
            # out = x.view(x.size(0), -1)

        return out, state_embedding

    def a3clstm(self, embedding, prev_hidden, params):
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:
            hx, cx = self._backend.LSTMCell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state # [1, 512, 7, 7]
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding # [1, 512, 7, 7]
        action_probs = model_input.action_probs # [7]
        params = model_options.params

        x, image_embedding = self.embedding(state, target, action_probs, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
