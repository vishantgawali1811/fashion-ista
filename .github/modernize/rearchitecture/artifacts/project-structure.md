# Project structure

## Project classification

Python deep-learning project with a static JavaScript frontend and local Python prediction API. It contains two related but distinct capabilities: offline Fashion-MNIST model training/evaluation and a browser demo backed by FashionCLIP zero-shot inference. It is not a database-backed application or a formal multi-package monorepo.

## Functional areas and layers

- **Dataset and shared ML utilities (`common.py`)**: reads local Fashion-MNIST CSVs, prepares image tensors, defines augmentation and compile/callback helpers, and writes evaluation artifacts.
- **Training experiments (`models/`)**: six separately runnable TensorFlow/Keras pipelines: CNN, VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0.
- **Prediction backend (`backend/server.py`)**: a threaded Python standard-library HTTP server that validates uploaded image data and uses a pretrained Hugging Face FashionCLIP model for zero-shot classification.
- **Browser UI (`frontend/`)**: standalone HTML/CSS/JavaScript page for selecting or dropping an image, previewing it, sending a request, and showing prediction/error states.
- **Local data (`data/`)**: Fashion-MNIST train and test CSV datasets.
- **Training output (`results/CNN/`)**: currently checked-in CNN model files, metrics, training history, and plots.
- **Dataset smoke script (`test_dataset.py`)**: loads the local dataset and prints split shapes, value ranges, labels, and class names.

## Runtime flow

1. `frontend/index.html` loads `config.js` and `app.js`; the default API URL is `http://127.0.0.1:5000/predict`.
2. A user selects/drops a JPG, PNG, or WEBP image. The browser previews it locally, enforces a 10 MB limit, then POSTs multipart form data with field `image`.
3. `backend/server.py` checks the request origin, content type, content length, upload size, MIME/file format, and pixel dimensions.
4. FashionCLIP image embeddings are compared with precomputed text prototypes derived from prompts for 18 categories. A low similarity or small lead produces `Other / unclear`.
5. The API returns JSON with a class, relative score, and ranked category matches; the page renders the result.

The training path is independent: each script in `models/` loads CSV data through `common.py`, trains/evaluates its own Keras classifier, and writes results under `results/<model>/`. The current `results/CNN/` artifacts do not supply the browser API model.

## Entry points and configuration

- `backend/server.py`: `python backend\server.py`; binds to `127.0.0.1:5000` by default; `THREADLINE_API_HOST` and `THREADLINE_API_PORT` override these values.
- `frontend/index.html`: static page served separately, for example with `python -m http.server 5500 --directory frontend`.
- `models/<model>.py`: each module has a `main()` guard and runs a training pipeline directly.
- `test_dataset.py`: root-level dataset verification script.
- `common.py`: shared train/test paths and class names are constants; uses relative `data/` and `results/` paths.

## Top-level layout

| Path | Role |
|---|---|
| `backend/` | FashionCLIP inference API |
| `frontend/` | Threadline static site |
| `models/` | Six Fashion-MNIST training scripts |
| `data/` | Local labeled Fashion-MNIST CSVs |
| `results/CNN/` | CNN model and evaluation outputs currently present |
| `common.py` | Shared training data and evaluation utilities |
| `test_dataset.py` | Dataset shape/label smoke check |
| `requirements.txt` | Python dependency names and version constraints |
| `.venv/` | Local Python 3.13 virtual environment; environment contents are not application source |
| `.vscode/tasks.json` | Workspace editor tasks |

## Source inventory

Excluding `.venv/`, `__pycache__/`, data files, and generated results: 9 Python source files (1 shared utility, 1 API server, 6 training scripts, 1 dataset smoke script) and 4 frontend files (1 HTML, 1 JavaScript application, 1 JavaScript config, 1 CSS stylesheet). The current `results/CNN/` folder contains 11 model/report/plot artifacts. The `data/` directory contains 2 CSV files.

No root-level application README, database schema, deployment configuration, or automated test suite was identified. `frontend/README.md` documents local frontend/API startup and its contract.
