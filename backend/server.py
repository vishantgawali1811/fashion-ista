"""Local fashion prediction API using FashionCLIP zero-shot classification."""

from __future__ import annotations

import json
import io
import os
import re
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_ID = "patrickjohncyh/fashion-clip"
CATEGORY_PROMPTS = {
    "Socks": (
        "a product photo of a pair of socks",
        "striped crew length socks on a white background",
        "a pair of ankle socks",
        "a pair of athletic socks",
        "a fashion photo of hosiery",
    ),
    "Sneakers": (
        "a product photo of sneakers",
        "a pair of athletic running shoes",
        "a product photo of sports shoes",
        "a pair of trainers",
        "a sneaker on a white background",
    ),
    "Sandals": (
        "a product photo of sandals",
        "a pair of open toe sandals",
        "a pair of flip flops",
    ),
    "Boots": (
        "a product photo of boots",
        "a pair of ankle boots",
        "a pair of winter boots",
    ),
    "Formal shoes": (
        "a product photo of formal leather shoes",
        "a pair of dress shoes",
        "a pair of loafers",
    ),
    "T-shirt/top": (
        "a product photo of a T-shirt",
        "a short-sleeve tee shirt",
        "a white crew neck T-shirt",
    ),
    "Shirt/blouse": (
        "a product photo of a button-up shirt",
        "a formal shirt or blouse",
        "a striped cotton shirt",
    ),
    "Sweater/pullover": (
        "a product photo of a sweater",
        "a pullover knitwear garment",
        "a knitted sweater",
    ),
    "Coat/jacket": (
        "a product photo of a coat",
        "a product photo of a jacket",
    ),
    "Dress": (
        "a product photo of a dress",
        "a one-piece fashion dress",
    ),
    "Skirt": (
        "a product photo of a skirt",
        "a knee length skirt",
    ),
    "Trousers": (
        "a product photo of trousers",
        "a pair of formal pants",
        "a pair of beige bootcut trousers",
        "a product photo of jeans",
    ),
    "Shorts": (
        "a product photo of shorts",
        "a pair of short pants",
    ),
    "Leggings": (
        "a product photo of leggings",
        "a pair of yoga leggings",
    ),
    "Bag": (
        "a product photo of a bag",
        "a handbag or purse",
        "a backpack",
    ),
    "Hat/cap": (
        "a product photo of a hat",
        "a baseball cap",
    ),
    "Scarf": (
        "a product photo of a scarf",
        "a fashion scarf",
    ),
    "Other clothing": (
        "a product photo of a fashion accessory",
        "a clothing product not included in the other categories",
    ),
}
CLASS_NAMES = tuple(CATEGORY_PROMPTS)
MINIMUM_CATEGORY_SIMILARITY = 0.22
MINIMUM_CATEGORY_MARGIN = 0.025
SOFTMAX_TEMPERATURE = 30.0
MAX_PROCESSING_SIZE = 1024
TEXT_MODEL_INPUTS = {
    "input_ids",
    "attention_mask",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def load_model() -> tuple[CLIPModel, CLIPProcessor, torch.Tensor]:
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
    model.to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    prompts = [
        prompt
        for category_prompts in CATEGORY_PROMPTS.values()
        for prompt in category_prompts
    ]
    prompt_inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = {
        name: value.to(DEVICE)
        for name, value in prompt_inputs.items()
        if name in TEXT_MODEL_INPUTS
    }
    with torch.inference_mode():
        text_output = model.get_text_features(**prompt_inputs)
        text_embeddings = (
            text_output.pooler_output
            if hasattr(text_output, "pooler_output")
            else text_output
        )
        text_embeddings = F.normalize(text_embeddings.float(), dim=-1)

    category_embeddings = []
    cursor = 0
    for category_prompts in CATEGORY_PROMPTS.values():
        next_cursor = cursor + len(category_prompts)
        category_embedding = text_embeddings[cursor:next_cursor].mean(dim=0)
        category_embeddings.append(F.normalize(category_embedding, dim=0))
        cursor = next_cursor

    prototypes = torch.stack(category_embeddings)
    print(
        f"Loaded {MODEL_ID} on {DEVICE}; "
        f"prepared {len(CLASS_NAMES)} fashion categories.",
        flush=True,
    )
    return model, processor, prototypes


MODEL, PROCESSOR, CATEGORY_EMBEDDINGS = load_model()


def preprocess_image(image_bytes: bytes) -> dict[str, torch.Tensor]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Choose a JPG, PNG or WEBP image.")
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large to process.")
            image = image.convert("RGB")
            image.thumbnail((MAX_PROCESSING_SIZE, MAX_PROCESSING_SIZE))
            image_inputs = PROCESSOR(
                images=image,
                return_tensors="pt",
            )
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Image dimensions are too large to process.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error

    return {
        name: value.to(DEVICE)
        for name, value in image_inputs.items()
    }


def _feature_tensor(model_output: torch.Tensor | object) -> torch.Tensor:
    features = (
        model_output.pooler_output
        if hasattr(model_output, "pooler_output")
        else model_output
    )
    if not isinstance(features, torch.Tensor):
        raise RuntimeError("FashionCLIP returned an unsupported feature format.")
    return features


def predict(image_bytes: bytes) -> dict:
    image_inputs = preprocess_image(image_bytes)
    with torch.inference_mode():
        image_output = MODEL.get_image_features(**image_inputs)
        image_embedding = F.normalize(_feature_tensor(image_output).float(), dim=-1)
        similarities = (
            image_embedding @ CATEGORY_EMBEDDINGS.T
        ).squeeze(0)

    if similarities.shape != (len(CLASS_NAMES),) or not torch.isfinite(similarities).all():
        raise RuntimeError("FashionCLIP returned invalid category scores.")

    category_order = torch.argsort(similarities, descending=True)
    best_index = int(category_order[0].item())
    best_similarity = float(similarities[best_index].item())
    second_similarity = float(similarities[category_order[1]].item())
    category_scores = torch.softmax(similarities * SOFTMAX_TEMPERATURE, dim=0)
    best_category = CLASS_NAMES[best_index]

    if (
        best_similarity < MINIMUM_CATEGORY_SIMILARITY
        or best_similarity - second_similarity < MINIMUM_CATEGORY_MARGIN
    ):
        best_category = "Other / unclear"

    return {
        "predicted_class": best_category,
        "confidence": float(category_scores[best_index].item()),
        "top_predictions": [
            {
                "label": CLASS_NAMES[int(index.item())],
                "confidence": float(category_scores[index].item()),
            }
            for index in category_order[:4]
        ],
    }


def get_image_part(content_type: str, body: bytes) -> tuple[str, bytes]:
    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("ascii", errors="strict")
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    if not message.is_multipart():
        raise ValueError("Upload an image using the image form field.")

    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") != "image":
            continue
        filename = part.get_filename() or "upload"
        image_bytes = part.get_payload(decode=True)
        media_type = part.get_content_type().lower()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")
        if media_type not in ALLOWED_IMAGE_TYPES | {"application/octet-stream"}:
            raise ValueError("Choose a JPG, PNG or WEBP image.")
        return filename, image_bytes

    raise ValueError("Choose an image in the image form field.")


class PredictionHandler(BaseHTTPRequestHandler):
    server_version = "ThreadlinePrediction/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)

    def send_json(self, status: HTTPStatus, data: dict) -> None:
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response)

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self._is_local_frontend_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    @staticmethod
    def _is_local_frontend_origin(origin: str) -> bool:
        if origin == "null":
            return True
        parsed = urlparse(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1"}
            and parsed.port is not None
        )

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": "FashionCLIP",
                    "categories": len(CLASS_NAMES),
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/predict":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return
        if not self._is_local_frontend_origin(self.headers.get("Origin", "null")):
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Request origin is not allowed."})
            return
        if not self.headers.get("Content-Type", "").lower().startswith("multipart/form-data;"):
            self.send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Upload an image using multipart form data."},
            )
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None or not re.fullmatch(r"\d+", content_length):
            self.send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "A valid content length is required."})
            return
        size = int(content_length)
        if size <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "The upload is empty."})
            return
        if size > MAX_UPLOAD_BYTES + 64 * 1024:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Image must be 10 MB or smaller."})
            return

        try:
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("The image upload was incomplete.")
            _, image_bytes = get_image_part(self.headers["Content-Type"], body)
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                self.send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "Image must be 10 MB or smaller."},
                )
                return
            self.send_json(HTTPStatus.OK, predict(image_bytes))
        except (ValueError, UnicodeError) as error:
            self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
        except Exception as error:
            print(f"Prediction failed: {error}", flush=True)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "The model could not analyze this image. Please try another image."},
            )


if __name__ == "__main__":
    host = os.environ.get("THREADLINE_API_HOST", "127.0.0.1")
    port = int(os.environ.get("THREADLINE_API_PORT", "5000"))
    server = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"Threadline prediction API listening at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Threadline prediction API.", flush=True)
    finally:
        server.server_close()
