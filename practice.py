from transformers import AlbertModel, AlbertTokenizer
import torch

# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertModel.from_pretrained('albert-base-v2')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  #


str1 = ["ADcfe", "djiEc"]
str2 = str1.copy()
for i, _ in enumerate(str2):
    str2[i] = str2[i].lower()

print(str1, str2)

s1 = "faeRF"
s2 = s1.lower()

print(id(s1), id(s2))
print(s1, s2)