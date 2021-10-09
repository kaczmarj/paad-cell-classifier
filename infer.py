"""Run inference using pre-trained pancreatic cell classifier."""

import click
from PIL import Image
import torch
from torch import nn
from torchvision import models
from torchvision import transforms


class LinearNetwork(nn.Module):
    def __init__(self, in_dim=1000, out_dim=200):
        super(LinearNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            # nn.Linear(1000, 1000),
            # nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            # nn.Softmax(dim = 1)
        )

    def forward(self, input1=None):

        # output1 = self.sim_model(input1)
        output2 = self.fc1(input1)

        return output2


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, lin_dim_in=1000, lin_dim_out=6):
        super(ResNet50, self).__init__()

        # define the resnet152
        resnet = models.resnet152(pretrained=pretrained)

        # isolate the feature blocks
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # average pooling layer
        self.avgpool = resnet.avgpool

        # classifier
        self.classifier1 = resnet.fc
        self.classifier2 = LinearNetwork(in_dim=lin_dim_in, out_dim=lin_dim_out)

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x, track_grads=False):

        # extract the features
        out_x = self.features(x)

        # register the hook
        if track_grads:
            _ = out_x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(out_x)
        # print("check the shape")
        # print(x.shape)
        # print(out_x.shape)
        # x = x.view((1, -1))
        x = x.view((x.shape[0], -1))
        # print("check the shape")
        # print(x.shape)
        x = self.classifier1(x)
        x = self.classifier2(x)
        if track_grads:
            return x, out_x
        else:
            return x


def load_image(path) -> torch.Tensor:
    tform = transforms.Compose(
        [
            transforms.Resize(280),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    p = Image.open(path)
    p = p.convert("RGB")
    p = tform(p)
    return p


@click.command()
@click.argument("image", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default="paad-classifier.pth",
)
def main(image, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(lin_dim_out=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    for image_path in image:
        image_tensor = load_image(path=image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        probs = nn.functional.softmax(output, dim=-1).squeeze().to("cpu")
        probs = probs.numpy()
        probs = dict(lymphocyte=probs[0], stroma=probs[1], tumor=probs[2])
        click.echo(probs)


if __name__ == "__main__":
    main()
