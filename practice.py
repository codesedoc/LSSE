from transformers import AlbertModel, AlbertTokenizer
import torch
import numpy as np
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertModel.from_pretrained('albert-base-v2')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  #


# str1 = ["ADcfe", "djiEc"]
# str2 = str1.copy()
# for i, _ in enumerate(str2):
#     str2[i] = str2[i].lower()
#
# print(str1, str2)
#
# s1 = "xsfaeRFxs"
# s2 = 'XXX'
#
# print(id(s1), id(s2))
# print(s1, s2)
# print(s1.strip()+s2)
# print(s1.strip('xs')+s2)

x = torch.tensor([-1, 2, -3]*3)
y = torch.abs_(x)
# # y[0] = 10
print(y)
#
print(x)

c = np.array(y.detach().numpy())
# c = y.detach().numpy()
# c = y.detach().numpy()
c [1:5] =0
print(c)
k = torch.tensor([1.3]).item()
print(y)
#
print(x)


#
#
# c[0] = 0
# print(y)
# y.sum().backward()
# print(x.grad)
# print(c)

# a = torch.tensor([1,2,3.], requires_grad =True)
# out = a.sigmoid()
# c = out.detach()
# c.zero_()

#
# out                   #  out的值被c.zero_()修改 !!
#
#
# out.sum().backward()
