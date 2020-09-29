---
layout: post

title: Python features you should be using

tip-number: 29
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn about features that will change the way you are writing Python code today
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Every new version of Python comes with tons of bug fixes and additional features. Here is a list of new features that you should be using

1. Enumerations - Python 3.4+
2. Data classes - Python 3.7+
3. Pathlib - Python 3.4+
4. Type hints - Python 3.5+
5. f-strings - Python 3.6+
6. Extended Iterable Unpacking - Python 3.0+
7. Walrus operator - Python 3.8+
8. Async IO - Python 3.4+
9. Underscores in Numeric Literals - Python 3.6+
10. LRU Cache - Python 3.2+

## 1. Enumerations - Python 3.4+
Instead of cluttering your code with constants, you can create an enumeration using the Enum class. An enumeration is a set of symbolic names bound to unique, constant values. 
```python
from enum import Enum

class Color(Enum):
  RED   = 1,
  GREEN = 2,
  BLUE  = 3

print(repr(Color.RED))
# <Color.RED: (1, )>

print(Color.GREEN)
# Color.GREEN

print(type(Color.BLUE))
# <enum 'Color'>
```

More information about enumerations:
* [https://docs.python.org/3.4/library/enum.html](https://docs.python.org/3.4/library/enum.html)
* [https://pythonspot.com/python-enum/](https://pythonspot.com/python-enum/)

## 2. Data classes - Python 3.7+

Using data classes, Python will automatically generate special methods like `_init__` and `__repr__`, reducing a lot of the clutter from your code.

This considerably reduces the amount of repetitive code that you had to write before.


```python
from dataclasses import dataclass

# Compare this class, using the conventional implementation
class Rectangle1:
  def __init__(self, color: str, width: float, height: float) -> None:
    self.color = color
    self.width = width
    self.height = height

  def area(self) -> float:
    return self.width * self.height

# with this class, using the @dataclass decorator
@dataclass
class Rectangle2:
  color: str
  width: float
  height: float

  def area(self) -> float:
    return self.width * self.height

# We can use both instances the same way
rect1 = Rectangle1("Blue", 2, 3)
rect2 = Rectangle2("Blue", 2, 3)

print(rect1)
# <__main__.Rectangle1 object at 0x7ff82020a100>

print(rect2)
# Rectangle2(color='Blue', width=2, height=3)

print(rect1.area())
# 6

print(rect2.area())
# 6
```

More information about data classes:
* [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)
* [https://docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)

## 3. Pathlib - Python 3.4+

The pathlib module provides a way to interact with the file system in a much more convenient way than dealing with os.path or the glob module.

The pathlib module makes everything simpler. When I discovered it, I've never looked back.
```python
from pathlib import Path

# This is a folder
path = Path("threads")
print(path)
# threads

# Let's make our path a little bit more complex
path = path / "sub" / "sub-sub"
print(path)
# threads/sub/sub-sub

# And now we can make the path absolute
print(path.resolve())
# /Users/dzlab/dev/twitter-threading/threads/sub/sub-sub
```

More information about the pathlib module:

* [https://realpython.com/python-pathlib/](https://realpython.com/python-pathlib/)
* [https://docs.python.org/3/library/pathlib.html](https://docs.python.org/3/library/pathlib.html)

## 4. Type hints - Python 3.5+

You can use type hints to indicate the type of a value in your code. For example, you can use it to annotate the arguments of a function and its return type.

These hints make your code more readable, and help tools understand it better.

```python
class Rectangle1:
  def __init__(self, width: float, height: float) -> None:
    self.width = width
    self.height = height

  def area(self) -> float:
    return self.width * self.height
```

More information about type hints:
* [http://veekaybee.github.io/2019/07/08/python-type-hints/](http://veekaybee.github.io/2019/07/08/python-type-hints/)
* [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)

## 5. f-strings - Python 3.6+

Instead of having to use the `.format()` method to print your strings, you can use f-strings for a much more convenient way to replace values in your strings.

f-strings are much more readable, concise, and easier to maintain.
```python
x = 2
y = 3

# Compare this statement:
print("x = {} and y = {}.".format(x, y))
# x = 2 and y = 3.

# with the f-string version
print(f"x = {x} and y = {y}.")
# x = 2 and y = 3.
```

More information about f-strings:

* [https://realpython.com/python-f-strings/](https://realpython.com/python-f-strings/)
* [https://python.org/dev/peps/pep-0498/](https://python.org/dev/peps/pep-0498/)

## 6. Extended Iterable Unpacking - Python 3.0+

Using this trick, while unpacking an iterable, you can specify a "catch-all" variable that will be assigned a list of the items not assigned to a regular variable.

Simple, but very convenient to keep the code concise.

Down pointing backhand index
```python
items = [1, 2, 3, 4, 5]
head, *body, tail = items

print(head, body, tail)
# 1 [2, 3, 4] 5
```

More information about Extended Iterable Unpacking:
* [https://python.org/dev/peps/pep-3132/](https://python.org/dev/peps/pep-3132/)
* [https://rfk.id.au/blog/entry/extended-iterable-unpacking/](https://rfk.id.au/blog/entry/extended-iterable-unpacking/)

## 7. Walrus operator - Python 3.8+

Using assignment expressions (through the walrus operator :=) you can assign and return a value in the same expression.

This operator makes certain constructs more convenient and helps communicate the intent of your code more clearly.

```python
# This is a regular while loop.
# Notice how we need to ask for a value twice.
value = input("Enter a value: ")
while value != "0":
  value = input("Enter a value: ")
  print(f"Value {value}")

# This is the same while loop, but now we are using
#  the walrus operator to avoid having to ask for the value twice.
while (value := input("Enter a value: ")) != 0:
  print(f"Value {value}")
```

More information about the Walrus operator:
* [https://deepsource.io/blog/python-walrus-operator/](https://deepsource.io/blog/python-walrus-operator/)
* [https://python.org/dev/peps/pep-0572/](https://python.org/dev/peps/pep-0572/)

## 8. Async IO - Python 3.4+

The asyncio module is the new way to write concurrent code using the `async` and `await` syntax.

This approach allows for much more readable code and abstracts away many of the complexity inherent with concurrent programming.

```python
import asyncio

async def say():
  print("Hello")
  await asyncio.sleep(1)
  print("World")

async def hello():
  await asyncio.gather(say(), say())

asyncio.run(hello())
# Hello
# Hello
# World
# World
```

More information about the asyncio module:
* [https://realpython.com/async-io-python/](https://realpython.com/async-io-python/)
* [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)

## 9. Underscores in Numeric Literals - Python 3.6+

This one is a small, nice addition: you can use underscores in numeric literals for improved readability.

This will shave off a few seconds every time you had to count how many digits a number had.
```python
x = 1_000_000
y = 1000000

print(x, y, x == y)
# 1000000 1000000 True
```

More information about underscores in numeric literals:
* [https://python.org/dev/peps/pep-0515/](https://python.org/dev/peps/pep-0515/)


## 10. LRU Cache - Python 3.2+

Using the `functools.lru_cache` decorator, you can wrap any function with a memoizing callable that implements a Least Recently Used (LRU) algorithm to evict the least recently used entries.

Do you want fast code? Look into this.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(number: int) -> int:
  if number == 0:
    return 0
  if number == 1:
    return 1
  return fib(number-1) + fib(number-2)

print(fib(50))
# 12586269025
```

More information about the `lru_cache` decorator:
* [https://docs.python.org/3/library/functools.html#functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)
* [https://cameronmacleod.com/blog/python-lru-cache](https://cameronmacleod.com/blog/python-lru-cache)

## Credits
https://twitter.com/svpino/status/1308632206278111232