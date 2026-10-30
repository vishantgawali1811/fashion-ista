# Fashion-Ista

Fashion-Ista is a local fashion inventory application with:

- A founder desk for classifying incoming clothing photos.
- A FashionCLIP-powered prediction API.
- A customer storefront with category filters, search, and a browser-based demo bag.
- Fashion-MNIST data and model-training artifacts in the `data`, `models`, and
  `results` folders.

## Requirements

- Python 3.10 or newer
- The packages listed in [`requirements.txt`](./requirements.txt)
- Internet access on the first API launch so Hugging Face can download the
  FashionCLIP model (approximately 605 MB)

Install dependencies from the project root:

```powershell
py -m pip install -r requirements.txt
```

## Run locally

Start the prediction API in one terminal:

```powershell
py backend\server.py
```

The API listens at <http://127.0.0.1:5000>. Verify it with:

<http://127.0.0.1:5000/health>

In a second terminal, serve the frontend:

```powershell
py -m http.server 5500 --directory frontend
```

Open the application:

- Founder desk: <http://127.0.0.1:5500/>
- Customer storefront: <http://127.0.0.1:5500/shop.html>

If ports 5000 or 5500 are already in use, choose alternate ports:

```powershell
$env:THREADLINE_API_PORT = "5001"
py backend\server.py
py -m http.server 5501 --directory frontend
```

When using an alternate API port, update `predictionUrl` in
[`frontend/config.js`](./frontend/config.js) to match it.

## How it works

The frontend sends an image as `multipart/form-data` to `/predict`. The API
uses FashionCLIP zero-shot classification and returns a predicted category,
relative confidence score, and the top category matches. Supported uploads are
JPG, PNG, and WEBP files up to 10 MB.

The founder desk stores inventory records and product images locally in the
browser. Products can include a name, quantity, category, price, color/size,
and delivery reference. The customer storefront reads those records from the
same browser and does not process payments or place real orders.

See [`frontend/README.md`](./frontend/README.md) for additional frontend and
API details.

## Tests

Run the dataset tests with:

```powershell
py -m pytest
```

## Repository

This project is maintained at
<https://github.com/vishantgawali1811/fashion-ista>.
