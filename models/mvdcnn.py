import torch
import torch.nn as nn
from torchvision import models


class MVDCNN(nn.Module):
    """
    Multi-View Convolutional Neural Network (MVCNN)
    Initializes a model with the architecture of a MVCNN with a ResNet34 base.
    """
    def __init__(self, premodel, model_name, num_classes):
        super(MVDCNN, self).__init__()
        base_model = premodel
        model_name = model_name

        if model_name == 'RESNET':
            fc_in_features = base_model.fc.in_features
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_in_features, num_classes)
                # nn.Dropout(),
                # nn.Linear(512, num_classes)
            )
        if model_name == 'VGG16':
        # for VGG-16 and a new FC
            fc_in_features = base_model.classifier[0].in_features

            self.features = nn.Sequential(*list(base_model.children())[:-1])
            # self.classifier = base_model.classifier
            # self.classifier[6] = torch.nn.Linear(in_features=base_model.classifier[3].out_features,
            #                               out_features=num_classes, bias=True)
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_in_features, num_classes)
                # nn.Dropout(),
                # nn.Linear(512, num_classes)
            )

    def forward(self, inputs): # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        # inputs = inputs.unsqueeze(0)
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)
            
        pooled_views = torch.mean(torch.stack(view_features), 0)
        # pooled_views, _ = torch.max(torch.stack(view_features), 0)
        # outputs = self.classifier(view_batch)
        outputs = self.classifier(pooled_views)
        return outputs