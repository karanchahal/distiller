import torch
import models
model_dict = {
    "WRN10_4": models.WRN10_4,  # params: 1198810
    "WRN16_1": models.WRN16_1,  # params: 175066
    "WRN16_2": models.WRN16_2,  # params: 691674
    "WRN16_4": models.WRN16_4,  # params: 2748890
    "WRN16_8": models.WRN16_8,  # params: 10961370
    "WRN28_2": models.WRN28_2,  # params: 1467610
    "WRN22_4": models.WRN22_4,  # params: 4298970
    "WRN22_8": models.WRN22_8,  # params: 17158106
    "WRN28_1": models.WRN28_1,  # params: 369498
    "WRN10_1": models.WRN10_1,  # params: 77850
    "WRN40_1": models.WRN40_1,  # params: 563930
    "WRN40_4": models.WRN40_4,  # params: 8949210
    # "resnet8": models.custom.resnet8_cifar,  # params: 78042
    "resnet14": models.custom.resnet14_cifar,  # params: 175258
    "resnet20": models.custom.resnet20_cifar,  # params: 272474
    "resnet32": models.custom.resnet32_cifar,  # params: 466906
    "resnet44": models.custom.resnet44_cifar,  # params: 661338
    "resnet56": models.custom.resnet56_cifar,  # params: 855770
    "resnet110": models.custom.resnet110_cifar,  # params: 1730714
    "resnet1202": models.custom.resnet1202_cifar,  # params: 19424026
    "resnet164": models.custom.resnet164_cifar,  # params: 1704154
    "resnet1001": models.custom.resnet1001_cifar,  # params: 10328602
    "resnet8": models.standard.ResNet8,  # params: 78042
    "resnet10": models.standard.ResNet10,  # params: 4903242
    "resnet18": models.standard.ResNet18,  # params: 11173962
    "resnet34": models.standard.ResNet34,  # params: 21282122
    "resnet50": models.standard.ResNet50,  # params: 23520842
    "resnet101": models.standard.ResNet101,  # params: 42512970
    "resnet152": models.standard.ResNet152,  # params: 58156618
    "vgg11": models.standard.VGG11,  # params: 9231114
    "vgg13": models.standard.VGG13,  # params: 9416010
    "vgg16": models.standard.VGG16,  # params: 14728266
    "vgg19": models.standard.VGG19,  # params: 20040522
    "efficientnet": models.standard.EfficientNetB0,  # params: 2912089
}


def create_cnn_model(name, num_classes, device):
    print(f"Building model {name}...")
    model_cls = model_dict[name]
    model = model_cls(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name} total parameters: {total_params}")
    # copy to cuda if activated
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    return model


# for model in model_dict.keys():
#     create_cnn_model(model, 10, "cpu")
