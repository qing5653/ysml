class Container:
    """A generic container that holds and manages a collection of items."""

    def __init__(self):
        self._items = []

    def add(self, item):
        """Add an item to the container."""
        self._items.append(item)

    def remove(self, item):
        """Remove an item from the container. Raises ValueError if not found."""
        self._items.remove(item)

    def get(self, index):
        """Return the item at the given index."""
        return self._items[index]

    def contains(self, item):
        """Return True if the item is in the container."""
        return item in self._items

    def size(self):
        """Return the number of items in the container."""
        return len(self)

    def is_empty(self):
        """Return True if the container has no items."""
        return not self._items

    def clear(self):
        """Remove all items from the container."""
        self._items.clear()

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"Container({self._items!r})"
