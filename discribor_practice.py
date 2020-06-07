class RevealAccess(object):
    """A data descriptor that sets and returns values
       normally and prints a message logging their access.
    """

    def __init__(self, initval=None, name='var'):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print(id(obj))
        print(obj)
        print(objtype)
        print('Retrieving', self.name)
        return self.val

    # def __set__(self, obj, val):
    #     print('Updating', self.name)
    #     self.val = val

class B(RevealAccess):
    ...

class A:
    x = B(1)
    def f(self):
        # self.x = 3
        print(self.x)

a = A()
# a.__dict__['x'] = 2
# a.x =2
a.f()
# print(A.x)
print(id(a))
print(a.__dict__)