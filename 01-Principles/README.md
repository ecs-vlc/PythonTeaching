# Principles to Python

This package of Jupyter notebooks covers the basics of the Python programming language. We only use in-built packages and the base Python library in this section, so there should be no dependencies.

There are questions at the end of each Jupyter notebook to test knowledge, with the complete solutions provided in the `solutions/` folder.

Internal packages we draw on include:

1. `itertools`
2. `re`
3. `os`
4. `sys`

## Basics of Python

In the first section, you will cover:

- The logic of Python
- Indentation
- Variables and Types
- Tuples
- Lists
- Basic Operators
- Operators with Lists
- String formatting
- String operations
- Conditions
- Advanced Operators
- Loops
- Functions
- Local and Global variables
- Classes and Objects
- Dictionaries
- Converting object types

We begin with a classical introduction to learning any programming language - saying hello to the world!

```python
print("Hello world!")
>>> Hello world!
```

Python is a non-staticly typed language, that is to say that the object *type* is not declared explicitly, but implicitly created for the programmer:

```python
my_int = 7
type(my_int)
>>> int
```

compared to a compiled-language such as **C**:

```C
int x = 4;
x = "four"; // FAILS
```

There are a number of powerful Python objects, such as **lists**, **tuples** and **dictionaries**:

```python
my_list = []
my_tuple = ()
my_dict = {}
```

Where these object store multiple values of different object types together. For loops, classes and functions, it is very important to have correct *indentation*, usually a tab character `\t` or 4 spaces:

```python
def my_square_function(x):
	return x*x
```

## Intermediate Features

The second section includes more advanced features of the basic platform, including:

- Sets
- List Comprehensions
- Reading/Writing Files
- With
- Exceptions
- Lambda/Reduce
- Arbitrary parameter lengths
- Decorators
- Internal functions

In addition to familiar structures in other programming languages such as for loops, Python has nice abstractions such as *list comprehensions*:

```python
list_comp = [i for i in range(10)]
print(list_comp)
>>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

this is the same as:

```python
list_comp = []
for i in range(10):
	list_comp.append(i)
print(list_comp)
>>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Sets are another incredibly powerful object, which are unique lists of elements (ints, strings) from which you can perform intersect and union-like operations.

A nice port from other scripting-style languages is the use of `lambda` and other operations:

```python
import math
f = lambda x: math.cos(x**2) - math.sin(x)
```

equivalent to creating a shorthand internal function. 

## Iterative tools

The third section includes the internal package `itertools`, which derives a number of useful **declarative** programming approaches to problems. It helps to overcome problems with the slow nature of Python *looping*. 

In-built Python bears the concept of an *iterator*, for example the `range(n)` object when created, doesn't actually instantiate `n` numbers, but rather creates an *iterator* from which numbers can be drawn in the execution on-the-fly. For example, let's say we want to cycle through every even number that exists, we can't do this procedurally due to the explosion in memory, but in declaration we can:

```python
import itertools as it
counter = it.count(start=0, step=2)
list(next(counter) for _ in range(5))
>>> [0, 2, 4, 6, 8]
```

They are also powerful for handling *recurrent relations*, since we can declare these *generator* objects such as `cycle` or `repeat`. For example we can encode the Fibonacci sequence trivially as:

```python
def fibs():
	a, b = 0, 1
	while True:
		yield a
		a, b = b, a + b

fibs = fibs()
```

Perhaps you want to generate all of the combinations between two vectors: `itertools` can return an *iterator* for you to go through this potentially massive list at your leisure:

```python
import itertools as it
ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"]
suits = ["H", "D", "S", "C"]
it.product(ranks, suits)
>>> <itertools.product object at 0x7fa360548f9438cd92>
```

And many more cumulative interactions between lists and other Python objects.

## Regular expressions

The fourth section includes the complex string manipulation library `re` or **Regex** for short. Regular expressions essentially act as a language for complex string manipulation and search. Regex strings are *compiled* into a series of bytecodes which are executed by a matching engine written in *C*. They are particularly useful for processing strings that have a known/inherent structure in them, for example postcodes, addresses, email addresses, URLs or HTML tags, for instance:

```python
import re
p = re.compile("[a-z]+")
print(p)
>>> re.compile(r'[a-z]+', re.UNICODE)
```

Will match one or more lowercase characters between 'a' and 'z', i.e all of them. 

## Project work

Here we provide a detailed hands-on example for users to attempt. No spoilers!

***

Inspirations: 
1. https://github.com/ageron/handson-ml
2. https://github.com/jakevdp/PythonDataScienceHandbook

***

The course is covered as interactive Jupyter notebooks which makes things considerably easier. In order to follow this course, we recommend you download Python using the Anaconda distribution (found [here](https://www.anaconda.com/download/)) as this also provides most of the packages used in this course.

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.

All rights reserved.