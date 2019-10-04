import torch.nn as nn
import torch.nn.functional as F

from fastai.vision import *
from fastai.vision.learner import cnn_config
from losses import ContrastiveLoss


class EmbeddingNet(nn.Module):
    def __init__(self, emsize, grayscale=False):
        super().__init__()
        input_ch = 1 if grayscale else 3
        self.convnet = nn.Sequential(
            nn.Conv2d(input_ch, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(1000000, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, emsize),
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


# class EmbeddingNetPretrained(nn.Module):
#     def __init__(self, pretrained_model_class, emsize):
#         super().__init__()
#         self.pretrained_model = create_cnn_model(pretrained_model_class, emsize)
#         if emsize < 256:
#             self.fc = nn.Sequential(
#                 nn.Linear(2048, 512),
#                 nn.PReLU(),
#                 nn.Linear(512, 256),
#                 nn.PReLU(),
#                 nn.Linear(256, emsize),
#             )
#         else:
#             self.fc = nn.Linear(2048, emsize)

#     def forward(self, x):
#         output = self.pretrained_model(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output

#     def get_embedding(self, x):
#         return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super().__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def siamese_learner(
    data,
    pretrained_model_class,
    emsize=128,
    margin=1.0,
    callback_fns=None,
):
    meta = cnn_config(pretrained_model_class)
    model = create_cnn_model(pretrained_model_class, emsize)
    model = SiameseNet(model)
    learn = Learner(
        data, model, loss_func=ContrastiveLoss(margin), callback_fns=callback_fns
    )
    learn.split(meta["split"](model.embedding_net))
    apply_init(model.embedding_net[1], nn.init.kaiming_normal_)
    return learn
