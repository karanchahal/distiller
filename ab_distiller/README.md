# Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons

Official Pytorch implementation of paper:

[Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/abs/1811.03233) (AAAI 2019).

Slides and poster are available on [homepage](https://sites.google.com/view/byeongho-heo/home)

## Environment
Python 3.6, Pytorch 0.4.1, Torchvision


## Knowledge distillation [(CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html) 

cifar10_AB_distillation.py

\
Distillation from WRN 22-4 (teacher) to WRN 16-2 (student) on CIFAR-10 dataset.

Pre-trained teacher network (WRN 22-4) is included. Just run the code.

## Transfer learning [(MIT_scenes)](http://web.mit.edu/torralba/www/indoor.html) 

MITscenes_AB_distillation.py 

\
Transfer learning from ImageNet pre-trained model (teacher) to randomly initialized model (student).

Teacher : ImageNet pre-trained ResNet 50

Student : MobileNet or MobileNetV2 (randomly initialized model)

Please change base learning rate to 0.1 for MobileNetV2.

\
MIT_scenes dataset should be arranged for Torchvision ImageFolder function.


Train set :
`$dataset_path / train / $class_name / $image_name `

Test set :
`$dataset_path / test / $class_name / $image name`


and run with dataset path.

MobileNet
```
python MITscenes_AB_distillation.py --data_root $dataset_path
```

MobileNet V2
```
python MITscenes_AB_distillation.py --data_root $dataset_path --network mobilenetV2
```
## Other implementations
Tensorflow: https://github.com/sseung0703/Knowledge_distillation_methods_wtih_Tensorflow

## Citation

```
@inproceedings{ABdistill,
	title = {Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons},
	author = {Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi},
	booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
	year = {2019}
}
```

