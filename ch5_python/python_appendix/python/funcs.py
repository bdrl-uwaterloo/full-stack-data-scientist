
# An example of function definition
def exponential (base,power):
    pass
# Call the exponential function 
exponential(2,4)

X = 'global'
def func():
    X = 'local'
func()
print(X)

def working_func():
    global X
    X = 'local'
working_func()
print(X)

# Argument passing:
aList = list(range(10))

def arg_demo (l):
    another_l = l
    print('l is aList? ', l is aList)
    print('another_l is aList', another_l is aList)
arg_demo(aList)

# TODO: arg_demo(tuple(aList) ) and arg_demo(aList[:])