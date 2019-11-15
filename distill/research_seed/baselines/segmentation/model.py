import torchvision.models.segmentation as seg 


model = seg.fcn_resnet50(pretrained=True)

# model2 = seg.fcn_resnet101(pretrained=True)
print(model2)