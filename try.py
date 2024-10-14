import timm
import torch
from torch import nn

# model = timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True, pretrained_cfg_overlay=dict(file="/home/suki/pycharm-professional-2024.2.3/checkpoints/vit/large/vit_large_patch14_clip_224.openai.bin"))
# model.head = nn.Linear(1024, 100)

print(timm.list_models())

# norm = nn.LayerNorm(normalized_shape=[3])
# a = [[[1, 2, 3], [0, 8, 3]], [[1, 2, 3], [0, 8, 3]]]
# b = [[1, 2], [2, 3], [0, 8]]
# a = torch.tensor(a)
# b = torch.tensor(b)
# c = a @ b
# print(c.size())
