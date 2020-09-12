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

**Useful magic commands**

| Command | Description |
|---------|-------------|
| `%%bash` | Make the accept shell commands instead of python code |
| `%%clear` | Clear cell output |
| `%debug` | Debug last execution that failed with an Exception |
| `%%writefile filename` | Write the cell content of into file (e.g. `%%writefile main.py`).|
