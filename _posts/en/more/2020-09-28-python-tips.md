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
* https://docs.python.org/3.4/library/enum.html
* https://pythonspot.com/python-enum/

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
* https://realpython.com/python-data-classes/
* https://docs.python.org/3/library/dataclasses.html

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

* https://realpython.com/python-pathlib/
* https://docs.python.org/3/library/pathlib.html

## Credits
https://twitter.com/svpino/status/1308632206278111232