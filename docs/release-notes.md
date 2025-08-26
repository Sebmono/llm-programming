---
title: Release Notes
description: Brief description of the thing
type: docs
weight: 7
---

## 2025-08-22 (v3.0.2)

### New Features
- Support direct injection of predicted output data into DSPy pipelines, ensuring user-provided predictions are forwarded to language model calls.
- Introduce `inspect_history` methods for language model modules to enable programmatic inspection of nested module histories.
- Enable Google Cloud Storage URL support (`gs://`) in the `dspy.Image` primitive for loading remote assets.
- Add a new `format` option to the `Image` primitive to customize output serialization.
- Expose a `dspy.configure_cache` function for fine-grained cache management and configuration.
- Allow authentication via Azure Databricks service principal (client ID/secret with token fallback) in the Databricks SDK client.

### Bug Fixes
- Fixed input key expansion in `DataLoader.from_parquet` to correctly unpack `input_keys` when creating `Example` instances.
- Corrected assignment of the language model instance in `ReinforceJob` constructor to store the provided LM object.
- Updated the MLflow deployment example to correctly access messages for the `forward` method in user-facing code snippets.
- Fixed handling of `None` values in usage tracking merges to ensure accurate aggregation of usage entries.
