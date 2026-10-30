# Threadline retail stock desk

A responsive, dependency-free boutique MVP with a founder stock desk and a
separate customer storefront. Founders can classify incoming clothing photos,
add product details, and publish in-stock pieces to the store. All frontend
files are contained in this folder.

## Run locally

The frontend and model API run in separate terminals. From the project root,
start the prediction API first:

```powershell
py backend\server.py
```

The API loads FashionCLIP from Hugging Face and listens at
<http://127.0.0.1:5000>. On first use, the model weights (about 605 MB) and
processor files are downloaded and cached; internet access is needed. In a
second terminal, serve the frontend:

```powershell
py -m http.server 5500 --directory frontend
```

Open the founder desk at <http://127.0.0.1:5500> or the customer storefront at
<http://127.0.0.1:5500/shop.html>. The API health check is
<http://127.0.0.1:5000/health>. Both servers use the existing Python
environment and project dependencies. Restart the API after changing backend
code so it loads the updated classifier.

`predictionUrl` is configured in [`config.js`](./config.js). The frontend sends
a `POST` request with `multipart/form-data` and an `image` file field. The API
returns a class label and category match score from `0` to `1`:

```json
{
  "predicted_class": "Sneaker",
  "confidence": 0.94,
  "top_predictions": [
    { "label": "Sneaker", "confidence": 0.94 },
    { "label": "Ankle boot", "confidence": 0.04 },
    { "label": "Sandal", "confidence": 0.02 }
  ]
}
```

FashionCLIP is a fashion-domain, zero-shot model. The API compares the image to
several descriptions for each of 18 categories: socks, sneakers, sandals, boots,
formal shoes, tops, shirts/blouses, sweaters, coats/jackets, dresses, skirts,
trousers, shorts, leggings, bags, hats/caps, scarves and other clothing. It
returns “Other / unclear” when the closest fashion category does not have a
strong enough match or lead. The score is a relative category match—not a
calibrated guarantee. The API accepts JPG, PNG and WEBP images up to 10 MB and
checks their decoded file format rather than trusting the upload MIME type.
The local API permits requests from localhost frontend origins.

## Retail stock log

The receiving screen reuses the existing single-image category prediction API.
After checking or editing a suggestion, add a product name, quantity, and
optional color/size, selling price, and delivery/slot reference. Products with
low-confidence or unclear category suggestions are held from `shop.html` until
the founder marks them checked. Other in-stock products appear on the
storefront automatically. The founder can search the stock log and remove
products.
The customer shop provides category filters, product search, and a local demo
bag capped by each product's available quantity. It does not accept payments,
place real orders, or reserve stock.

The MVP stores product records and the demo bag in browser local storage and
product photos in IndexedDB on the current device. Data does not sync across
devices or users. The match score is relative, not a calibrated confidence
guarantee; review product categories before relying on them for inventory
decisions.
