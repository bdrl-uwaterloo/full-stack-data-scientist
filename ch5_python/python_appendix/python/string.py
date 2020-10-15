# String literals in Python
single_quote = 'Hi from single quote'
print(single_quote)
double_quote = "double quote's greetings"
print(double_quote)
triple_quote = '''triple quote's
greetings can be
multiple lines'''
print(triple_quote)
# Concatenation
s = 'Hello' + 'Ivan'
print(s)
greeting = 'Hello'
name = 'Ivan'
s = greeting + name
print(s)
s = greeting + ',' + ' ' + name
print(s)
msg = '{}, {NAME}. Welcome to {}'.format(greeting,'Zeronelab',NAME=name)
print(msg)
# upper(), lower()
msg = 'Hello, Jordan'
print(msg.upper())
print(msg.lower())
print('The original message is: ' + msg)
print('Equality Check: ' + str(msg.lower() == msg.lower()))
print('Identity Check: ' + str(msg.lower() is msg.lower()))
# Use of find(sub) 
msg = 'Hello, Jordan'
print(msg.find(' '))
print(msg.find('o'))
print(msg.find('Jordan'))
print(msg.find('Ivan'))
# Use of find(sub, start)
msg = 'Hello, Jordan'
print(msg.find(',',5))
print(msg.find(',',6))
print(msg.find('o',4))
print(msg.find('o',5))
# Use of find(sub, start, end)
msg = 'Hello, Jordan'
print(msg.find('o',0,4))
print(msg.find('o',4,-1))
print(msg.find('o',5,-1))
# Use of count(sub) 
msg = 'Hello, Jordan'
print(msg.count(' '))
print(msg.count('o'))
print(msg.count('Jordan'))
print(msg.count('Ivan'))
# Use of count(sub, start)
msg = 'Hello, Jordan'
print(msg.count(',',5))
print(msg.count(',',6))
print(msg.count('o',4))
print(msg.count('o',5))
# Use of count(sub, start, end)
msg = 'Hello, Jordan'
print(msg.count('o',0,4))
print(msg.count('o',4,-1))
print(msg.count('o',5,-1))
# replace()
msg = "example 1 example 2 example 3 example 4"
print(msg.replace('example','apple'))
print(msg.replace('ex',''))
print(msg.replace('exam','ap',2))
# split() with one or two optional
msg = 'First,Second,Third,Fourth'
print(type(msg.split(',')))
returned_list = msg.split(',')
print(returned_list)
returned_list = msg.split(',', 2)
print(returned_list)
# consecutive delimiters
msg = 'First,Second,,Third'
print(msg.split(','))
# 'Ivan' splits a sentence
msg = 'FirstIvanSecondIvanThirdIvanFourth'
print(msg.split('Ivan'))
# split(sep = None) vs split(' ') 
msg = 'First Second   Third    Fourth'
print(msg.split())
print(msg.split(' '))
# str.join()
delimiter = ','
iterable = 'Ivan'
print(delimiter.join(iterable))
# The type of the iterable does not matter
iterable = list('Ivan')
print(list('Ivan'))
print(delimiter.join(iterable))
print(type(delimiter.join(iterable)))
# But the type of all items must be str
iterable = ['0',1,2,'3',4,5]
print(delimiter.join(iterable))
# Slicing a String
s = '0123456789'
# Print the first digit in the string
print(s[0])
# Print the first three digits in the string
print(s[:3])
# Print the substring from the fourth digit onwards
print(s[3:])
# Step 1: slices are copies:
s = '0123456789'
print(s[3:] is s[3:])
# Step 2: they are shallow copies:
s1 = s[1:5]
print(s[1] is s1[0])
print(s[2] is s1[1])
print(s[3] is s1[2])
print(s[4] is s1[3])
s = '0123456789'
# Only elements with even indices:
# starting at 0 and skips 1 elements per step
print(s[::2])
# Only elements with odd indices:
# starting at 1 and skips 1 elements per step
print(s[1::2])
# Only elements with odd indices that devides 3:
# starting at 0 and skips 2 elements per step
print(s[0::3])
# negative indices
s = '0123456789'
print(len(s))
print(s[-1])
print(s[-4:-1])
print(s[-4+len(s):-1+len(s)])
msg = 'reverse me!'
print(msg[::-1])
# str.startswith()
s = 'aString1'
print(s.startswith('aS'))
print(s.startswith('aSt',0))
print(s.startswith('a',1))
print(s.startswith('aSt',0,2))
print(s.startswith('aSt',0,-2))
# str.endswith()
s = 'aString1'
print(s.endswith('g1'))
print(s.endswith('ng1',0))
print(s.endswith('a',1))
print(s.endswith('g1',0,-1))
print(s.endswith('aSt',0,-2))