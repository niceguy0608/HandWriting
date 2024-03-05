import torch
a=torch.arange(32).reshape(2,4,4)

b=torch.max(a,0)
print(b)

