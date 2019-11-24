from research_seed.baselines.model.model_factory import create_cnn_model, is_resnet
import torch 
from rkd_baseline import addEmbedding
import argparse 

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('--output-size', default=64, type=int, help='batch_size')
parser.add_argument('--embedding-size', default=128, type=int, help='batch_size')
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')


hparams = parser.parse_args()

model = create_cnn_model('resnet110', feature_maps=True)

a = torch.randn((1,3,32,32))

pool, b1, b2, b3 = model(a)

for m in model.modules():
    m.requires_grad = False


embed_model = addEmbedding(model, hparams)

for m in embed_model.linear.modules():
    print(m.requires_grad)

# print(pool.size())
# print(b1.size())
# print(b2.size())
# print(b3.size())



