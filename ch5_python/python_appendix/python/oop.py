
# Probably the simplest class possible
class people():
    pass

# the name people is a class object
print(type(people))

# An instance of people is of type people or __main__.people
Ivan = people()
print(type(Ivan))

# In Java, the following raises compiler error of multiple definitions
class people():
    print('Hey, this is another person')

people()

# An example of multiple inheritance
class sup1():
    pass

class sup2():
    pass

class sub(sup1, sup2):
    pass

# Example of constructor
class student():
    def __init__(self, name, std_number):
        self.name = name
        self.std_number = std_number

ivan = student('ivan','20942325')
print(ivan.name,ivan.std_number)

# TODO more examples on class variables.