#
# x1 = 0
def testt():
    x = 0

    class C:
        def text(self):
            print("cadas2")
        for i in range(10):
            ...
        print(x)
        print(locals())

    def t():
        # print(x)
        print(x1)
        print(locals())

    # import practice as p
    # p.text()
    print(id(C))

    x1 = x
    # del x
    print(x1)
    # print(x)
    print(locals())
    t()

    del C.text
    C.text=1
    C.ts =0
    print(C.text)
    print(C.__dict__)

    # del p
    # print(p)
if True:
    class C:
        x = 0
        def text():
            # print(x)
            print("trtretr")
        for i in range(10):
            ...
        text()
        # print(locals())

testt()
# C = 0
C.text()
C.tdst = testt
print(C().tdst)
print(C.x)
print((((((i*i for i in range(10)) ))), 1))
print([])
# p.text()
# testt.t()