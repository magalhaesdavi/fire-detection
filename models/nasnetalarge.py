import pretrainedmodels
import torch.nn as nn

# print(pretrainedmodels.model_names)
# print(pretrainedmodels.pretrained_settings['nasnetalarge'])

def get_model(nb_classes):
    model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # model.eval()

    # fine tuning
    dim_feats = model.last_linear.in_features # =2048
    # nb_classes = 2
    model.last_linear = nn.Linear(dim_feats, nb_classes)
    # output = model(input_224)
    # print(output.size())
    return model
