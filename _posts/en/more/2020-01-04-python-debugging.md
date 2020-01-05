---
layout: post

title: Debugging in Python

tip-number: 00
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: My training loop just crashed and I have no idea what the error does mean, what should I do?
tip-writer-support: https://www.patreon.com/dzlab

redirect_from:
  - /en/python-debugging/

categories:
    - en
    - more
---

Python is by far the most used lanuguage for Deep Learning and mastering debugging in Python is very important as it can save you headackes when trying to identigy problems in your code.

Just like any language, Python has an interactive debugger called `pdb`. To use it, simply put a `pdb.set_trace()` right in the place where something weird is happening. During execution, the program will be halted in this breakpoint and you will be able to access current state (i.e. variables).

```python
import pdb

# something weird is happening here and we need to debug
pdb.set_trace()
```

Another way to use the debugger, especially in jupyter notebooks, is through the magic command `%debug` to trace an error. Use it just after a python exception had occured and an interactive debugging session will be started with the progam state the moment the exception happened. Very cool, isn't it?

Once a debugging session is started, you can use the following commands and their shortcuts:

| Command        | Short version           | Comment  |
| ------------- |:-------------:| -----:|
| `help` | `h`| for help |
| `step` | `s`| to step inside the current instruction. |
| `next` | `n`| simply type enter to run next instruction. |
| `continue`| `c`| to continue until next breakpoint. |
| `up`   | `u`| to change the context of the debugger and see what the previous call on stack, i.e. what called the current instruction.|
| `down` | `d`| (after a `u`) to go down agin and return to the previous debugger context.|
| `print`| `p`| to print a variable, e.g. `p my_variable`.|
| `list` | `l`| for listing the code context, i.e. lines before and after current instruction.|

More debugging tips can be found [here](https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger).
