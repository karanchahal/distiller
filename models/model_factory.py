import torch
import cifar10
import cifar10sm
import vision
import wide_resnet as wide


model_dict = {
    "WRN10_1": wide.WRN10_1,  # params: 77850 Cifar10: 0.8806
    "WRN16_1": wide.WRN16_1,  # params: 175066 Cifar10: 0.9175
    "WRN28_1": wide.WRN28_1,  # params: 369498 Cifar10: 0.9289
    "WRN40_1": wide.WRN40_1,  # params: 563930 Cifar10: 0.9372
    "WRN16_2": wide.WRN16_2,  # params: 691674  Cifar10: 0.9418
    "WRN28_2": wide.WRN28_2,  # params: 1467610 Cifar10: 0.9489
    "WRN10_4": wide.WRN10_4,  # params: 1198810 Cifar10: 0.9236
    "WRN16_4": wide.WRN16_4,  # params: 2748890 Cifar10: 0.9542
    "WRN22_4": wide.WRN22_4,  # params: 4298970 Cifar10: 0.9584
    "WRN40_4": wide.WRN40_4,  # params: 8949210  Cifar10: 0.9589
    "WRN16_8": wide.WRN16_8,  # params: 10961370 Cifar10: 0.9606
    "WRN22_8": wide.WRN22_8,  # params: 17158106 Cifar10: 0.9616
    # 3 layer weak cifar resnets
    "resnet8_sm": cifar10sm.resnet8,  # params: 78042
    "resnet14_sm": cifar10sm.resnet14,  # params: 175258
    "resnet20_sm": cifar10sm.resnet20,  # params: 272474
    "resnet32_sm": cifar10sm.resnet32,  # params: 466906
    "resnet44_sm": cifar10sm.resnet44,  # params: 661338
    "resnet56_sm": cifar10sm.resnet56,  # params: 855770
    "resnet110_sm": cifar10sm.resnet110,  # params: 1730714
    "resnet1202_sm": cifar10sm.resnet1202,  # params: 19424026
    "resnet164_sm": cifar10sm.resnet164,  # params: 1704154
    "resnet1001_sm": cifar10sm.resnet1001,  # params: 10328602
    # 3 layer cifar resnets
    "resnet8": cifar10.resnet8,  # params: 89322  Cifar10: 0.89590
    "resnet14": cifar10.resnet14,  # params: 186538 Cifar10: 0.92180
    "resnet20": cifar10.resnet20,  # params: 283754 Cifar10: 0.93020
    "resnet26": cifar10.resnet26,  # params: 380970 Cifar10: 0.92180
    "resnet32": cifar10.resnet32,  # params: 478186 Cifar10: 0.93690
    "resnet44": cifar10.resnet44,  # params: 672618 Cifar10: 0.94400
    "resnet56": cifar10.resnet56,  # params: 867050 Cifar10: 0.94510
    # 4 layer cifar resnets
    "resnet10": cifar10.resnet10,  # params: 4903242 Cifar10: 0.94300
    "resnet18": cifar10.resnet18,  # params: 11173962 Cifar10: 0.95260
    "resnet34": cifar10.resnet34,  # params: 21282122
    "resnet50": cifar10.resnet50,  # params: 23520842
    "resnet101": cifar10.resnet101,  # params: 42512970
    "resnet152": cifar10.resnet152,  # params: 58156618
    # torchvision resnets designed for imagenet
    "resnet8_v": vision.resnet8,  # params: 95082
    "resnet14_v": vision.resnet14,  # params: 192298
    "resnet20_v": vision.resnet20,  # params: 289514
    "resnet10_v": vision.resnet10,  # params: 4910922
    "resnet18_v": vision.resnet18,  # params: 11181642
    "resnet34_v": vision.resnet34,  # params: 21289802
    "resnet50_v": vision.resnet50,  # params: 23528522
    "resnet101_v": vision.resnet101,  # params: 42520650
    "resnet152_v": vision.resnet152,  # params: 58164298
    "wrn50_2": vision.wide_resnet50_2,  # params: 66854730
    "wrn101_2": vision.wide_resnet101_2,  # params: 124858186
    # vgg lol
    "vgg11": cifar10.VGG11,  # params: 9231114
    "vgg13": cifar10.VGG13,  # params: 9416010
    "vgg16": cifar10.VGG16,  # params: 14728266
    "vgg19": cifar10.VGG19,  # params: 20040522
}


def create_model(name, num_classes, device):
    model_cls = model_dict[name]
    print(f"Building model {name}...", end='')
    model = model_cls(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    layers = len(list(model.modules()))
    print(f" total parameters: {total_params}, layers {layers}")
    # always use dataparallel for now
    model = torch.nn.DataParallel(model)
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPU(s).")
    # copy to cuda if activated
    model = model.to(device)
    return model


if __name__ == "__main__":
    for model in model_dict.keys():
        create_model(model, 10, "cpu")
