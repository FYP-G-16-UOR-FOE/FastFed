# FastAPI minimal setup

This workspace contains a minimal FastAPI example and step-by-step instructions to create a Python virtual environment (venv) on macOS (zsh) and run the app.

Files added:

- `main.py` - minimal FastAPI app
- `requirements.txt` - pip dependencies
- `tests/test_main.py` - small pytest-based test (see below)

How to create a Python environment (recommended using venv)

1. Create the venv (uses the system Python or any python3 in PATH):

```bash
python3 -m venv .venv
```

2. Activate the venv (zsh):

```bash
source .venv/bin/activate
```

3. Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the FastAPI app (development):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the automatic Swagger UI.

Testing

With the venv activated and deps installed, run:

```bash
pytest -q
```

Notes

- If you prefer conda, create an environment with `conda create -n myenv python=3.11` and activate it with `conda activate myenv`, then install with `pip install -r requirements.txt`.
- `.venv` is a common name; add it to `.gitignore` if you commit.
