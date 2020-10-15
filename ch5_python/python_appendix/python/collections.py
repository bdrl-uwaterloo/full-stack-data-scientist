# Create an empty list:
my_list = []
my_list = list()
# Create a list of 10 one's
my_list = [1]*10
print(my_list)
# Concatenation:
print([1]*5 + [2] * 5)
print([1,2,3,4] + ['one','two','three','four'])
# Slicing
l = [1,2,3,4,5]
# start index is inclusive
# end index is exclusive
print(l[1:3])
# if I want the last two elements
print(l[-2:])
# slicing with a step
print(l[::2])
# quickly reverse a list
print(l[::-1])
# List comprehension
a = list(range(5))
# basic list comprehension example 1
a_squared = [n ** 2 for n in a ]
print('example 1: ', a_squared)
# basic list comprehension example 2
a_odd_element = [n for n in a if n%2==1]
print('example 2: ', a_odd_element)
# basic list comprehension example 3
a_parity = ['odd' if i%2==1 else 'even' for i in a]
print('example 3: ', a_parity)
# more complicated list comprehension example
x_y_tuple = [ (x,y) for x in range(1,6) if x%2 == 1 for y in range(7,11) if y%2 == 0]
print('example 4: ', x_y_tuple)
# Create an empty tuple:
my_tuple = ()
my_tuple = tuple()
# Tuples are immutable
my_list = [1,2,3,4]
my_tuple = (1,2,3,4)
my_list[0] = 0
my_tuple[0] = 0
# Set is unsorted
my_set = {'1,2,3,4', 'Ivan', 'Jordan'}
print(my_set)
my_set.add(5)
print(my_set)
print(my_set)