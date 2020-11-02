---
layout: post

title: Colab tips

tip-number: 26
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to use Colab efficiently?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

**Prevent Colab from disconnecting**

On Chrome, open console via `More tools` -> `Developer Tools`
```javascript
function keepAwake(){
  console.log("Awake");
  document.querySelector("#comments > span").click();
}
setInterval(keepAwake, 5000);
```

**Hide code cell**

Simply add a title to the Code cell (e.g. `#@title cell title`) or

1. Right-click on the area on the left of the cell (below the "Play" button) and choose "Add a form"
2. Enter a title for the cell like `#@title cell title`
3. Right-click again in the same place and choose "Form > Hide code"

**Collapsible markdown**

<details><summary>CLICK</summary>
<p>

#### hidden code block in a text cell!

```python
print("hello world!")
```

</p>
</details>

**Useful magic commands**

| Command | Description |
|---------|-------------|
| `%%bash` | Make the accept shell commands instead of python code |
| `%%clear` | Clear cell output |
| `%debug` | Debug last execution that failed with an Exception |
| `%%writefile [-a] filename` | Write the cell content of into file (e.g. `%%writefile main.py`). If `-a`/`--append` is provided then the content of the cell will be appended to the file.|
| `%%file filename` | alias for `%%writefile`.|
