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

# x = torch.tensor([-1, 2, -3]*3)
# y = torch.abs_(x)
# if torch.isnan(y).sum()>0:
#     print(x)
# # y[0] = 10
# print(y)
# #
# print(x)
#
# c = np.array(y.detach().numpy())
# # c = y.detach().numpy()
# # c = y.detach().numpy()
# c [1:5] =0
# print(c)
# k = torch.tensor([1.3]).item()
# print(y)
# #
# print(x)
#
# torch.optim.lr_scheduler.LambdaLR
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
# class c:
#     value =1
#     pass
# x =dict({1:[2],2:5,4:c()})
# t = list(x.values())
# print(x[4])
# print(t[2])
# print(x[4].value)
# print(t[2].value)
# t[2].value = 0
# print(x[4].value)
# print(t[2].value)

# import tqdm
# import time
# for i in tqdm.trange(5, desc="Epoch"):
#     for j in tqdm.tqdm(range(2), desc="iteration"):
#         time.sleep(0.5)
# def func(c1, c2=3):
#     print(c1, c2)
# func(2, 1,)
import logging

# root = logging.root
logging.debug('logger debug message')
logging.info('logger info message')
logging.warning('logger warning message')
logging.error('logger error message')
logging.critical('logger critical message')

# logging.basicConfig(
#         format=" %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level= logging.DEBUG,
#     )

# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level= logging.INFO,
#     )

logger1 = logging.getLogger('mylogger')
# logger1.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('./test.log')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger1.addHandler(fh)
logger1.addHandler(ch)


logger1.propagate = False
logger1.debug('logger1 debug message')
logger1.info('logger1 info message')
logger1.warning('logger1 warning message')
logger1.error('logger1 error message')
logger1.critical('logger1 critical message')

pass