import torchvision.models as models
from torch import nn


def build_model(num_classes , base = "vgg"):
    if base=='vgg':
        return vgg_model(base, num_classes)
class vgg_model(nn.Module):
    def __init__(self, base, num_classes):
        super(vgg_model, self).__init__()
        self.model = models.vgg16()
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class efficient_model(nn.Module):
    def __init__(self, base, num_classes):
        super(efficient_model, self).__init__()
        self.model = models.vgg16()
        # self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__=="__main__":
    print(build_model(5))


