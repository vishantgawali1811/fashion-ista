# Existing technology stack

## Languages and runtimes

- Python for data preparation, TensorFlow model training, and the prediction API.
- HTML, CSS, and browser JavaScript for the frontend; no frontend package manager or framework is used.
- The local `.venv/pyvenv.cfg` records CPython 3.13.0. The project does not declare a Python version requirement in `requirements.txt`.

## Frameworks and libraries

Declared in `requirements.txt`:

| Dependency | Version declaration |
|---|---|
| `tensorflow` | Unpinned |
| `numpy` | Unpinned |
| `pandas` | Unpinned |
| `scikit-learn` | Unpinned |
| `matplotlib` | Unpinned |
| `seaborn` | Unpinned |
| `Pillow` | Unpinned |
| `torch` | `>=2.2,<3` |
| `transformers` | `>=4.40,<6` |

Usage observed in source: TensorFlow/Keras for supervised CNN and ImageNet transfer-learning experiments; PyTorch and Hugging Face Transformers (`CLIPModel`, `CLIPProcessor`) for FashionCLIP; NumPy/Pandas for CSV data; scikit-learn metrics; Matplotlib/Seaborn for reports and plots; Pillow for upload decoding and image preprocessing. The API uses Python's standard-library `http.server`, `email`, `json`, and `urllib.parse` modules.

Dependency versions are not comprehensively locked. No `pyproject.toml`, lockfile, setup metadata, or explicit Node.js dependency manifest was found in the project root. The local virtual environment is not a reproducible dependency lock.

## Build and runtime configuration

- TensorFlow training scripts use eager Keras model APIs; transfer-learning backbones use ImageNet weights and may download weights when first loaded.
- FashionCLIP model ID is `patrickjohncyh/fashion-clip`; weights and processor are downloaded/cached by Hugging Face on first startup, requiring network access.
- The API chooses CUDA when available and otherwise CPU.
- Frontend fonts are fetched from Google Fonts; frontend source assets otherwise have no build step.
- Python server defaults: host `127.0.0.1`, port `5000`; environment variables `THREADLINE_API_HOST` and `THREADLINE_API_PORT` can override them.
- Runtime data and output paths in `common.py` are relative (`data/`, `results/`), so running training scripts from the project root is expected.

## Runtime and model specifics

- Fashion-MNIST images are stored as 28×28 grayscale pixels and converted to float32 values in `[0,1]`, reshaped to 28×28×1, then replicated to RGB (28×28×3).
- Transfer-learning models resize/preprocess inputs in their Keras models; InceptionV3 uses a 299×299 input and the other transfer-learning scripts use 224×224.
- The browser API accepts JPG/JPEG, PNG, and WEBP; max upload is 10 MiB and max decoded dimensions are 20 million pixels.
- Browser and backend are separately hosted local processes; CORS responses are allowed for local HTTP origins with an explicit port and the `null` origin.

## Deprecated/EOL signals

No explicit deprecated or end-of-life API usage could be established from the dependency manifest. Version ranges for TensorFlow and most numerical/visualization dependencies are absent, so compatibility and support status cannot be determined from this repository alone.
