import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello, FastAPI!"}


def test_read_item():
    resp = client.get("/items/42?q=things")
    assert resp.status_code == 200
    assert resp.json() == {"item_id": 42, "q": "things"}
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello, FastAPI!"}


def test_read_item():
    resp = client.get("/items/42?q=things")
    assert resp.status_code == 200
    assert resp.json() == {"item_id": 42, "q": "things"}
