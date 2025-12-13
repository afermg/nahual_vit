# Load model directly
import numpy
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "recursionpharma/OpenPhenom", trust_remote_code=True, dtype="auto"
).cuda()
# %%
tile_size = 256
data = numpy.random.random_sample((1, 6, tile_size, tile_size))
torch_tensor = torch.from_numpy(data).float().cuda()
result = model.predict(torch_tensor)
print([x.shape for x in result])
# torch.Size([1, 384])
np_val = result.cpu().detach().numpy()
