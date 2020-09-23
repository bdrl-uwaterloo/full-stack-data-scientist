# if statement:
aNumber = 3
if aNumber >= 0:
    if aNumber == 0:
        print("first string")
else:
    print("second string")
print("third string")
# elif
aNumber = int(input('Please input a number: '))
if aNumber > 0:
    print("{} is greater than 0 ".format(aNumber))
elif aNumber == 0:
    print("{} is equal to 0 ".format(aNumber))
else:
    print("{} is less than 0 ".format(aNumber))
# for loop
for i in range(10):
    print(i, end=" ")
for letter in 'Ivan and Jordan':
    print(letter, end="")
aList = [1,2,3,4]
for i in aList:
    if i % 2:
        print('odd')
    else:
        print ('even')
else:
    print('checked all elements in {}'.format(aList))
aList = [1,2,3,4]
for i in aList:
    if i % 2:
        print('odd')
    else:
        print ('even')
        break # Note the break statement here!
else:
    print('checked all elements in {}'.format(aList))
# working with zip
first = ['Ivan', 'Alan', 'Jordan', 'Eric']
last = ['Zeng', 'Walker', 'Zeng', 'Peng']
names = zip(first,last)
emails = []
for first_name, last_name in names:
    emails.append('{}{}@email.com'.format(first_name,last_name))
emails
# while loop
user_input = int(input('Please enter a number: '))
while user_input:
    if user_input < 0:
        print('invalid input please try again')
        user_input = int(input('Please enter a number: '))
    else:
        i = user_input // 2
        while i > 1:
            if user_input % i == 0:
                print('{} divides {}'.format(i,user_input))
                user_input = int(input('Please enter a number: '))
                break
            else:
                i -= 1
        else:
            print('{} is a prime number'.format(user_input))
            user_input = int(input('Please enter a number: '))
else:
    print('exit with user input 0')