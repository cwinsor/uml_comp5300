
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
# from bigram import BigramLanguageModel

assert False, "hold up"

modelpath = r'models/m0304_213215_state.cpk'

model = BigramLanguageModel()
model.load_state_dict(torch.load(modelpath))

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.zeros((1, 1), dtype=torch.long)

# print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print( model.generate(context, max_new_tokens=100)[0].tolist() )
