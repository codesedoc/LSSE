# # # from transformers import AlbertModel, AlbertTokenizer
# # # # import torch
# # # # import numpy as np
# # # #
# # # #
# # # # # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# # # # # model = AlbertModel.from_pretrained('albert-base-v2')
# # # # # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# # # # # outputs = model(input_ids)
# # # # # last_hidden_states = outputs[0]  #
# # # #
# # # #
# # # # # str1 = ["ADcfe", "djiEc"]
# # # # # str2 = str1.copy()
# # # # # for i, _ in enumerate(str2):
# # # # #     str2[i] = str2[i].lower()
# # # # #
# # # # # print(str1, str2)
# # # # #
# # # # # s1 = "xsfaeRFxs"
# # # # # s2 = 'XXX'
# # # # #
# # # # # print(id(s1), id(s2))
# # # # # print(s1, s2)
# # # # # print(s1.strip()+s2)
# # # # # print(s1.strip('xs')+s2)
# # # #
# # # # # x = torch.tensor([-1, 2, -3]*3)
# # # # # y = torch.abs_(x)
# # # # # if torch.isnan(y).sum()>0:
# # # # #     print(x)
# # # # # # y[0] = 10
# # # # # print(y)
# # # # # #
# # # # # print(x)
# # # # #
# # # # # c = np.array(y.detach().numpy())
# # # # # # c = y.detach().numpy()
# # # # # # c = y.detach().numpy()
# # # # # c [1:5] =0
# # # # # print(c)
# # # # # k = torch.tensor([1.3]).item()
# # # # # print(y)
# # # # # #
# # # # # print(x)
# # # # #
# # # # # torch.optim.lr_scheduler.LambdaLR
# # # # #
# # # # #
# # # # # c[0] = 0
# # # # # print(y)
# # # # # y.sum().backward()
# # # # # print(x.grad)
# # # # # print(c)
# # # #
# # # # # a = torch.tensor([1,2,3.], requires_grad =True)
# # # # # out = a.sigmoid()
# # # # # c = out.detach()
# # # # # c.zero_()
# # # #
# # # # #
# # # # # out                   #  out的值被c.zero_()修改 !!
# # # # #
# # # # #
# # # # # out.sum().backward()
# # # # # class c:
# # # # #     value =1
# # # # #     pass
# # # # # x =dict({1:[2],2:5,4:c()})
# # # # # t = list(x.values())
# # # # # print(x[4])
# # # # # print(t[2])
# # # # # print(x[4].value)
# # # # # print(t[2].value)
# # # # # t[2].value = 0
# # # # # print(x[4].value)
# # # # # print(t[2].value)
# # # #
# # # # # import tqdm
# # # # # import time
# # # # # for i in tqdm.trange(5, desc="Epoch"):
# # # # #     for j in tqdm.tqdm(range(2), desc="iteration"):
# # # # #         time.sleep(0.5)
# # # # # def func(c1, c2=3):
# # # # #     print(c1, c2)
# # # # # func(2, 1,)
# # # # # import logging
# # # #
# # # # # root = logging.root
# # # # # logging.debug('logger debug message')
# # # # # logging.info('logger info message')
# # # # # logging.warning('logger warning message')
# # # # # logging.error('logger error message')
# # # # # logging.critical('logger critical message')
# # # #
# # # # # logging.basicConfig(
# # # # #         format=" %(levelname)s - %(name)s -   %(message)s",
# # # # #         datefmt="%m/%d/%Y %H:%M:%S",
# # # # #         level= logging.DEBUG,
# # # # #     )
# # # #
# # # # # logging.basicConfig(
# # # # #         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
# # # # #         datefmt="%m/%d/%Y %H:%M:%S",
# # # # #         level= logging.INFO,
# # # # #     )
# # # #
# # # # # logger1 = logging.getLogger('mylogger')
# # # # # # logger1.setLevel(logging.DEBUG)
# # # # # # 创建一个handler，用于写入日志文件
# # # # # fh = logging.FileHandler('./test.log')
# # # # # # 再创建一个handler，用于输出到控制台
# # # # # ch = logging.StreamHandler()
# # # # # # 定义handler的输出格式formatter
# # # # # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # # # # fh.setFormatter(formatter)
# # # # # ch.setFormatter(formatter)
# # # # #
# # # # # logger1.addHandler(fh)
# # # # # logger1.addHandler(ch)
# # # # #
# # # # #
# # # # # logger1.propagate = False
# # # # # logger1.debug('logger1 debug message')
# # # # # logger1.info('logger1 info message')
# # # # # logger1.warning('logger1 warning message')
# # # # # logger1.error('logger1 error message')
# # # # # logger1.critical('logger1 critical message')
# # # # class O:
# # # #     def __init__(self):
# # # #         print('O')
# # # #         super().__init__()
# # # #         pass
# # # #
# # # # class O1:
# # # #     def __init__(self):
# # # #         print('O1')
# # # #         pass
# # # #
# # # #
# # # #
# # # #
# # # #
# # # # class B:
# # # #     def __init__(self):
# # # #         print('B')
# # # #         super().__init__()
# # # #
# # # #
# # # # class C(O, O1):
# # # #     def __init__(self):
# # # #         print('C')
# # # #         super(B, self).__init__()
# # # #
# # # #
# # # # class C1(C):
# # # #     def __init__(self):
# # # #         print('C1')
# # # #         # print('C1 mro:{}'.format((C1.mro())))
# # # #         super().__init__()
# # # #         # super().__init__()
# # # #         # super().__init__()
# # # #
# # # #
# # # # # class D(B, A):
# # # # #     def __init__(self):
# # # # #         print('D')
# # # # #         # a = C()
# # # # #         super().__init__()
# # # # #         # super().__init__()
# # # # #         # print('---------------')
# # # # #         # super().__init__()
# # # #
# # # #
# # # # class E(B):
# # # #     def __init__(self):
# # # #         print('E')
# # # #         # a = C()
# # # #         super().__init__()
# # # #         # super().__init__()
# # # #         # print('---------------')
# # # #
# # # class F():
# # #     def __init__(self):
# # #         print('F')
# # #         # a = C()
# # #         super().__init__()
# # #         # super().__init__()
# # #         # print('---------------')
# # #
# # #     def __get__(self, instance, owner):
# # #         print(instance, owner)
# # #         print("getxxxxx")
# # #
# # #     # def __set__(self, instance, value):
# # #     #     print("setxxxxx")
# # # #
# # # # print('mro:{}'.format(F.mro()))
# # # # class A:
# # # #     f = F()
# # # #     def __init__(self):
# # # #         print(__class__)
# # # #         print('A mro:{}'.format((A.mro())))
# # # #         print('A')
# # # #         self.f = 1
# # # #         # super(O,self).__init__()
# # # #
# # # #     # def __getattribute__(self, item):
# # # #     #     print("xfdfsd")
# # # # #
# # # # # f = F()
# # # # a = A()
# # # # A.f = f
# # # # # a.a = f
# # # # print(a.__dict__)
# # # # # a.a
# # #
# # # # print(A.f)
# # # # print(type(a).__dict__['f'].__get__(a, type(a)))
# # # # print(a.f)
# # # # print(A.f.__dict__)
# # # # # print(A.f)
# # # # # print(A.__dict__)
# # # # # print(a.__dict__)
# # # # # a = A()
# # # # # import builtins
# # # # # # if __name__
# # # # # # print(__dict__)
# # # # # print(builtins.__dict__['__file__'])
# # # # # super(D, f).__init__()
# # # # # import tmp
# # # # # import model
# # # # # from model import gcn
# # # # # a = tmp.__path__
# # # # # print(a)
# # # # pass
# # #
# # # #
# # # class MyProperty(object):
# # #     "Emulate PyProperty_Type() in Objects/descrobject.c"
# # #
# # #     def __init__(self, fget=None, fset=None, fdel=None, doc=None):
# # #         self.fget = fget
# # #         self.fset = fset
# # #         self.fdel = fdel
# # #         if doc is None and fget is not None:
# # #             doc = fget.__doc__
# # #         self.__doc__ = doc
# # #
# # #     def __get__(self, obj, objtype=None):
# # #         if obj is None:
# # #             return self
# # #         if self.fget is None:
# # #             raise AttributeError("unreadable attribute")
# # #         return self.fget(obj)
# # #
# # #     def __set__(self, obj, value):
# # #         if self.fset is None:
# # #             raise AttributeError("can't set attribute")
# # #         self.fset(obj, value)
# # #
# # #     # def __delete__(self, obj):
# # #     #     if self.fdel is None:
# # #     #         raise AttributeError("can't delete attribute")
# # #     #     self.fdel(obj)
# # #
# # # class A:
# # #     def __init__(self, name, score):
# # #         self.name = name  # 普通属性
# # #         self.score = score
# # #
# # #     def getscore(self):
# # #         return self._score
# # #
# # #     def setscore(self, value):
# # #         print('setting score here')
# # #         # print(self.score)
# # #         if isinstance(value, int):
# # #             self._score = value
# # #             pass
# # #         else:
# # #             print('please input an int')
# # #
# # #     score = MyProperty(getscore, setscore)
# # #
# # # #
# # #
# # # a = A('Bob', 90)
# # # # a.__dict__ = {}
# # # print(a.__dict__)
# # # # A.score=2
# # # # =
# # # # a.d = MyProperty(A.getscore, A.setscore)
# # # # print(A.d )
# # # # print(a._A__score)
# # #
# # # # print()
# # # print(A.score)
# # # b = A('Bob', 20)
# # # print(b.score)
# # # print(a.score)
# # #
# # # print(A.score)
# # # # a.name  # 'Bob'
# # # # a.score  # 90
# # # # print(a.score)
# # # # a.score = 'bob'  # please input an int
#
# def func(name, d=0, age=0, *, sex):
#     pass
# # def func(**k):
# #     print(k)
#     # sex[2] = 'sdfsd'
#     # print(sex)
#
# func('tanggu',100,[2]*10, sex= [2]*10)
# # func('tanggu',2,)
# # func('tanggu',2,)
#
# # class A:
# #     pass
# # class ContextM:
# #     pass
# #     # def __enter__(self):
# #     #     print('enter')
# #
# #     def __exit__(self, exc_type, exc_val, exc_tb):
# #         print('exit')
# #         print("type: ", exc_type)
# #         print("val: ", exc_val)
# #         print("tb: ", exc_tb)
# #         return True
# # print(dir(A))
# # print(dir(ContextM))
# # with ContextM():
# #     1/0
# #     # raise ValueError('fdsfa')
# #     pass
#


def Singleton(cls):
    _instance = None
    def _singleton(*args, **kargs):
        nonlocal _instance
        if _instance == None:
            _instance = cls(*args, **kargs)
        return _instance
    print('id{}'.format(id(_singleton)))
    return _singleton


# @Singleton

class A():
    """Docstring"""

    def __init__(self , a):
        self.a = "aa"
        print(a)

class B():
    """Docstring"""

    def __init__(self , a):
        self.b = "bb"
        print(a)

A1 = A


print(type(A), type(B))

A = Singleton(A)
d = Singleton(B)


print(A, d)
print(id(A), id(d))
print(A.__closure__)
print(d.__closure__)
a1 = A(1)
a2 = d(2)

print(A.__closure__)
print(d.__closure__)

a1 = A(1)
a2 = d(2)

print(a1, a2)


def func(a):
    aa = 0
    return aa

x = func(1)
