# Architecture artifact index

## Implementation Guide

This index is not the full contract. Do not implement from this file alone; follow the artifact paths below.

### Global artifacts

- `unit_graph.yaml`
  - Who reads: all workers.
  - Use for: unit triggers, source anchors, externally visible signatures, dependencies, and shared references.
  - How to consume: locate the assigned unit; anchors are evidence of behavior, not a rewrite scope.
- `shared_modules.yaml`
  - Who reads: workers using shared training code.
  - Use for: common helper names, types, consumers, and handling guidance.
  - How to consume: filter `modules` by `used_by_units`; each unit's `shared_refs.used_fields` is the allowed subset.
- `wire_contracts.yaml`
  - Who reads: UI and API workers.
  - Use for: request/response and error contracts.
  - How to consume: select rows whose `unit` equals the assigned unit.
- `cross_unit_state.yaml`
  - Who reads: all workers when changing state across entrypoints.
  - Use for: implicit session/static/request state flows.
  - How to consume: currently `flows: []`; no cross-unit implicit state flow was identified.
- `project-structure.md`, `tech-stack.md`, and `data-model.md`
  - Who reads: all workers.
  - Use for: project boundaries/layers, existing dependencies/runtime, and data/persistence facts.

No `migration_boundary.yaml` is included because no rewrite/migration acceptance criteria were supplied. No `seams.yaml` is included because no partial-migration cut point was declared or needed for the inspected current architecture.

## Implementation Guide by unit

### `threadline-ui`

- External trigger: load the static page and select/drop an image.
- Must read:
  - `units/threadline-ui/behavior.yaml` — upload validation, local preview lifecycle, request and result/error states.
  - `units/threadline-ui/bindings.yaml` — API URL runtime configuration.
  - `units/threadline-ui/unit_decomposition.yaml` — advisory split candidates only.
- Relevant global rows: `wire_contracts.yaml` where `unit: threadline-ui`; `unit_graph.yaml` for its API dependency.
- Before DONE report: artifacts read, preserved user-visible flows/contracts, unresolved items, and browser/runtime evidence.

### `fashion-prediction-api`

- External trigger: `GET /health`, `POST /predict`, or CORS `OPTIONS`.
- Must read:
  - `units/fashion-prediction-api/behavior.yaml` — model loading, image processing/classification, branches and HTTP errors.
  - `units/fashion-prediction-api/bindings.yaml` — endpoint declarations and host/port configuration.
  - `units/fashion-prediction-api/unit_decomposition.yaml` — advisory split candidates only.
- Relevant global rows: `wire_contracts.yaml` where `unit: fashion-prediction-api`; `unit_graph.yaml` for public routes.
- Before DONE report: artifacts read, implemented API/error contracts, unresolved items, and request/runtime evidence.

### `train-fashion-mnist-cnn`

- External trigger: `python models/cnn.py`.
- Must read: `units/train-fashion-mnist-cnn/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: `shared_modules.yaml` where `used_by_units` contains this unit; `unit_graph.yaml` for the helper subset.
- Before DONE report: artifacts read, preserved dataset/training/evaluation outputs, unresolved items, and training/test evidence.

### `train-fashion-mnist-vgg16`

- External trigger: `python models/vgg16.py`.
- Must read: `units/train-fashion-mnist-vgg16/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for helper subset.
- Before DONE report: artifacts read, preserved two-phase fine-tuning and outputs, unresolved items, and training/test evidence.

### `train-fashion-mnist-resnet50`

- External trigger: `python models/resnet50.py`.
- Must read: `units/train-fashion-mnist-resnet50/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for helper subset.
- Before DONE report: artifacts read, preserved two-phase fine-tuning and outputs, unresolved items, and training/test evidence.

### `train-fashion-mnist-mobilenetv2`

- External trigger: `python models/mobilenetv2.py`.
- Must read: `units/train-fashion-mnist-mobilenetv2/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for helper subset.
- Before DONE report: artifacts read, preserved two-phase fine-tuning and outputs, unresolved items, and training/test evidence.

### `train-fashion-mnist-inceptionv3`

- External trigger: `python models/inceptionv3.py`.
- Must read: `units/train-fashion-mnist-inceptionv3/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for helper subset.
- Before DONE report: artifacts read, preserved 299×299 preprocessing, two-phase fine-tuning and outputs, unresolved items, and training/test evidence.

### `train-fashion-mnist-efficientnetb0`

- External trigger: `python models/efficientnetb0.py`.
- Must read: `units/train-fashion-mnist-efficientnetb0/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for helper subset.
- Before DONE report: artifacts read, preserved two-phase fine-tuning and outputs, unresolved items, and training/test evidence.

### `test-fashion-mnist-dataset`

- External trigger: `python test_dataset.py`.
- Must read: `units/test-fashion-mnist-dataset/{behavior,bindings,unit_decomposition}.yaml`.
- Relevant global rows: shared-module row where `used_by_units` contains this unit; `unit_graph.yaml` for its loader/class-name subset.
- Before DONE report: artifacts read, preserved dataset validation/printed checks, unresolved items, and execution evidence.
