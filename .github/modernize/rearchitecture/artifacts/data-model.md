# Existing data and model

## Persistence

No database, ORM, entity classes, or persistent request store is present. The project uses local CSV files as the supervised training dataset. The prediction API handles image bytes in memory and returns a JSON response; it does not persist uploaded images or prediction history in the inspected source.

## Fashion-MNIST CSV schema

Both `data/fashion-mnist_train.csv` and `data/fashion-mnist_test.csv` have 785 columns: one `label` column plus 784 pixel columns representing a 28×28 grayscale image. Files contain 60,000 training records and 10,000 test records (excluding their header rows). Pixel values are normalized by division by 255; labels are integer class IDs 0–9.

The loader in `common.py` takes the last `validation_size` records from the training CSV as validation (default 5,000), leaving 55,000 training records. It does not apply a separate shuffle/split operation at load time. Test records remain in the test set.

| Label | Class name |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Browser inference JSON contract

The UI POSTs `multipart/form-data` with an `image` file part to `/predict`. Successful API responses contain:

- `predicted_class`: one of the 18 FashionCLIP category labels or `"Other / unclear"`.
- `confidence`: the selected category's normalized relative score, in `[0,1]` (the API's top-category score, including when abstaining).
- `top_predictions`: up to four `{label, confidence}` objects for the highest-scoring categories.

The score is a relative category match and is not a calibrated probability. The API scores 18 named categories; the `"Other / unclear"` abstention is a decision outcome and is not one of the embedding categories.

## Training model outputs

The supervised training classifier uses ten Fashion-MNIST classes. `results/CNN/` currently includes Keras `best_model.keras` and `final_model.keras`, a model summary, training history, classification report, raw and normalized confusion matrices, and plots. The output schema is artifact files rather than a database.

The CNN classification report records 90.84% accuracy over 10,000 test records; weighted precision is approximately 0.90836, weighted recall 0.9084, and weighted F1 approximately 0.90807.

## Key entities/records

- **Fashion-MNIST example**: one integer category label and 784 grayscale pixel values; the primary training/evaluation record.
- **Prediction upload**: transient image bytes extracted from the multipart `image` part; not persisted.
- **Prediction result**: JSON response with class, relative score, and ranked category list.
- **Training history record**: per-epoch accuracy/loss, validation accuracy/loss, and learning rate in `training_history.csv`.
- **Evaluation report**: per-class precision, recall, F1, and support plus aggregate scores in `classification_report.csv`.
