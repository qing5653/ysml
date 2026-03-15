# ysml

A generic container implementation in Python.

## Container

`Container` is a simple, iterable collection that holds an ordered list of items.

### Usage

```python
from container import Container

c = Container()

# Add items
c.add("apple")
c.add("banana")

# Check membership
c.contains("apple")   # True

# Access by index
c.get(0)              # "apple"

# Iterate
for item in c:
    print(item)

# Size
c.size()              # 2
len(c)                # 2

# Remove
c.remove("apple")

# Clear all
c.clear()

# Check empty
c.is_empty()          # True
```

### Running Tests

```bash
python -m pytest tests/
```