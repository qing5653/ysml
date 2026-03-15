import pytest
from container import Container


def test_new_container_is_empty():
    c = Container()
    assert c.is_empty()
    assert c.size() == 0


def test_add_item():
    c = Container()
    c.add("apple")
    assert c.size() == 1
    assert c.contains("apple")


def test_add_multiple_items():
    c = Container()
    c.add(1)
    c.add(2)
    c.add(3)
    assert c.size() == 3


def test_remove_item():
    c = Container()
    c.add("apple")
    c.remove("apple")
    assert c.is_empty()


def test_remove_nonexistent_item_raises():
    c = Container()
    with pytest.raises(ValueError):
        c.remove("missing")


def test_get_item_by_index():
    c = Container()
    c.add("first")
    c.add("second")
    assert c.get(0) == "first"
    assert c.get(1) == "second"


def test_contains_returns_false_when_absent():
    c = Container()
    assert not c.contains("ghost")


def test_clear_empties_container():
    c = Container()
    c.add(1)
    c.add(2)
    c.clear()
    assert c.is_empty()


def test_iteration():
    c = Container()
    items = [10, 20, 30]
    for item in items:
        c.add(item)
    assert list(c) == items


def test_len():
    c = Container()
    c.add("x")
    c.add("y")
    assert len(c) == 2


def test_repr():
    c = Container()
    c.add(1)
    assert repr(c) == "Container([1])"
