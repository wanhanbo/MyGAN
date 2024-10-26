class A:
    def getinfo(self, x,y,z):
        p=x+y+z
        a=p*(p-1)
        print("t1的结果",a)
    def getinfo(self, x,y):
        a=x*y
        print("t2的结果",a)
    def getinfo(self,x):
        a=2*x
        print("t3的结果",a)
a1=A().getinfo(2)
a1=A().getinfo(3,4)
a1=A().getinfo(3,4,5)