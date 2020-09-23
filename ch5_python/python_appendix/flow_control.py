# if statement:
aNumber = 3
if aNumber >= 0:
    if aNumber == 0:
        print("first string")
else:
    print("second string")
print("third string")
# There is no else if for example, the following won't run:

#if x>3:
#    pass
#else if:
#    pass

# Instead, one should use the keyword elif
if x>3:
    pass
elif y<2:
    pass

# For loop similar to for each loop

aList = [1,2,3,4,5,'Ivan','Leah','Alan']
bList = aList[:]

print("aList == bList: " , aList == bList)
print("aList is bList: " , aList is bList)


for item in aList:
    print(item)

# For loop as ordinary loop

for i in range(1,10,2):
    print(i)

# Quick intro to boolean
x = list(range(1,10))

for i in x:
    if not i:
        print(i)
        break
else:
    print("all true")

# While loop test if integer x is a prime number

y = 99
x = y/2

# The above example won't work!!

# Use the floor division

y = 99
x = y//2 + 1

while x:
    if y%x == 0:
        print(x, ' divides ', y)
        break
    x -= 1
else:
    print(y, 'prime number')