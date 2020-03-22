---
layout: post

title: Python __dunder__ thingies

tip-number: 22
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: What's python data model?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

As part of Python data model, a set of special methods on objects (which are also called `__dunder__`) are provided to perform special operations. To learn more on python data model check the documentation - [link](https://docs.python.org/3/reference/datamodel.html).

## Object life-cycle
In some cases, the object needs to perform some action and manage its state when it's created or deleted. This can be done by implementing following methods.
```python
# to do something when the object is deleted, i.e. del(o)
def __del__(self, ...)
# to initialize the object
def __init__(self, ...)
# to do something when a new object instance is called
def __new__(self, ...)
```

## Attribute management
Set and get object attributes by implementing following methods:
```python
# to get value of an attribute of this object
def __getattr__(self, ...)
# to set value of an attribute of this object
def __setattr__(self, ...)
```
As a result, we can set and get the attributes of an object as follows:
```python
>>> setattr(o, 'attribute', 1)
>>> getattr(o, 'attribute')
1
```
Also to check if an object has a specific attribute, use `hasattr` as follows:
```python
hasattr(obj, 'attribute_or_function_name')
```

## Collection management
It is possible to make an object behave like a collection by implementing following methods.
```python
# to get an item by index, i.e. obj[i]
def __getitem__(self, ...)
# to get len of the object (e.g. if it contains a collection)
def __len__(self, ...)
```
Now the object can be used as follows:
```python
>>> len(o)
10
>>> o[1]
'Element at position 1'
```

## Context management
Context management is useful when dealing with resources (e.g. open file). i,e. some logic has to be executed when entering the context (e.g. opening a file) and later when leaving the context (e.g. closing file and cleaning resources). In python, we can support such behavior simply by implementing the following methods:
```python
# to do something when the context of the object is entered (i.e. with obj: )
def __enter__(self, ...)
# to do something when the context of the object is existed
def __exit__(self, ...)
```
Now the object can be used as follows:
```python
with o:
  # do something with object 'o'
```

## Arithmetic operations
We can augment an object to support basic arithmetic operations (e.g. addition) with following methods
```python
# to add support for + operation
def __add__(self, ...)
```
Now we can do:
```python
>>> o1 + o2
```

## Object representation
To change the default string representation of an object so that to return useful information on its state rather than default memory address, implement following methods:
```python
# to modify the string representation
def __repr__(self, ...)
# to add support for formatted print (i.e. str(o))
def __str__(self, ...)
```

## Misc
To make an object callable implement:
```python
# to be able to call the object
def __call__(self, ...)
```
For example
```python
>>> o(param1)
```