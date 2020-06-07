class O:
    def __init__(self):
        print('O')

        super().__init__()
        pass

class O1:
    ooo = 2
    def test(self):
        print('test')
    def __init__(self):
        print(self)
        print('O1')
        pass





class B:
    def __init__(self):
        print('B')
        super().__init__()


class C(O, O1):
    ooo = 3
    def __init__(self):
        print('C')
        super(O, self).__init__()


class C1(C):
    def __init__(self):
        print('C1')
        # print('C1 mro:{}'.format((C1.mro())))
        super().__init__()
        # super().__init__()
        # super().__init__()


# class D(B, A):
#     def __init__(self):
#         print('D')
#         # a = C()
#         super().__init__()
#         # super().__init__()
#         # print('---------------')
#         # super().__init__()


class E(B):
    def __init__(self):
        print('E')
        # a = C()
        super().__init__()
        # super().__init__()
        # print('---------------')

def aaa():
    print('AAAA')
class F(C):
    _super = super(O)
    def __init__(self):
        print('F')
        # self.fff = a
        # a = C()
        # super.aaa = aaa
        # print(super(F).__init__(O, self))
        # print(super(F).__class__.mro())
        # print(super(F, self).__init__)
        super(F, self).__init__()
        print('**************')
        # super(O, self).__init__()
        self._super.__init__()
        # super().__init__()
        # super().__init__()
        # print('---------------')

    # def __get__(self, instance, owner):
    #     print(instance, owner)
    #     print("getxxxxx")
    #
    # def __set__(self, instance, value):
    #     print("setxxxxx")

# print('mro:{}'.format(F.mro()))
# class A:
#     f = F()
#     def __init__(self):
#         print(__class__)
#         print('A mro:{}'.format((A.mro())))
#         print('A')
#         self.f = 1
#         super(O,self).__init__()

    # def __getattribute__(self, item):
    #     print("xfdfsd")
# F.
f = F()
print(C.mro())
# print(type(None))
# print(super(None.__class__, None).__init__)
# print(super(F, None))
# print(type(super(F, None)))
#
# print(super(F, f))
# print(type(super(F, f)))
#
# print(F.__init__)
# print(f.__init__)
# print(type(f.__init__))
# print(type(f.__dir__))
#
# print(super(F, F).__init__)
print(F._super)
print(f._super)
print("*******")

print(F._super.__get__(F).__init__)
print(F._super.__get__(f).__init__)

print("***********")
print(F._super.__get__(F))
print(F._super.__get__(C))
print(F._super.__get__(f, C))
print(F._super.__get__(C,f))
print(F._super.__get__(None, C))
print(F._super.__get__(None, f))
# print(O1.__init__)
# print(C.ooo)
# print(F.__dict__)
print(super(F,f))
print(super(F,F))
# print(super(F,f).test)
# print(super(C,f).test())
# a = A()
# A.f = f
# # a.a = f
# print(a.__dict__)
# # a.a

# print(A.f)
# print(type(a).__dict__['f'].__get__(a, type(a)))
# print(a.f)
# print(A.f.__dict__)
# # print(A.f)
# # print(A.__dict__)
# # print(a.__dict__)
# # a = A()

