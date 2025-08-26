---
title: Release Notes
description: Brief description of the thing
type: docs
weight: 7
---

## 2025-07-23 (v3.0.0b3)
### New Features
- **BAMLAdapter for Structured Outputs**: Added a BAML-compatible adapter for structured outputs with readable, compact Pydantic schema rendering and field descriptions embedded as comments, enabling better reasoning about complex output types.
- **DSPy Code Type**: Introduced `dspy.Code` type with language parameter for language-aware code blocks in signatures and module inputs/outputs.
- **XMLAdapter Streaming Support**: XMLAdapter streaming now fully enabled and unit-tested with correct field and chunk boundary detection.
- **Trace and History Size Limiting**: Added `max_history_size` and `max_trace_size` settings to globally control the retention of module and trace histories to prevent memory overflow in long-running programs.
- **StreamListener Reuse**: StreamListeners now support optional `allow_reuse` for use with ReAct and other repeated-field streaming scenarios.

### Enhancements
- **Code Serialization**: Improved serialization and deserialization logic to better handle non-primitive and Pydantic types (e.g., datetime, HttpUrl) by always serializing in valid JSON.
- **BAMLAdapter Robustness**: BAMLAdapter now handles deeply nested, aliased, circular, and mixed Pydantic models robustly with more human-usable output.
- **Token-Efficient Schema Rendering**: BAMLAdapter's schema rendering is more token efficient for small models compared to JSONAdapter.
- **Optional Field Handling**: BAMLAdapter and base types handle `Optional` and union types with correct presentation and type casting.
- **Improved Custom Type Parsing**: Better fallback handling for Type and custom annotation extraction for Python 3.11/3.10 compatibility.
- **Consistent LM and Module History Retention**: All histories and traces now follow retention limits set via DSPy context.

### Bug Fixes
- **History and Trace Memory Leak**: Trace and history objects are now automatically capped and kept up to a maximum as per user settings to avoid unbounded memory growth in iterative or parallel scenarios.
- **Custom Type Message Splitting**: Fixed incorrect message splitting when custom types use single quotes (Python repr) by attempting parsing as valid JSON before falling back to alternate approaches.
- **Pydantic List/Dict Nested Schema Output**: Correct bracket notation and recursion for list[Model] fields in BAMLAdapter schemas.
- **Forward Usage Warning**: Added warning and improved detection when a user mistakenly calls `forward()` directly instead of using instance `program()`. 
- **Parse Type Handling in Adapter**: Ensured adapters do not raise on parse when target type is Union or contains None or custom types.
- **Dataset Dependency Only Where Needed**: Hasher now vendored to avoid hard dependency on `datasets` for most DSPy usage.
- **Random/Trace/History Truncation in Predict/Module**: Fixes bug where LM and prediction histories build indefinitely, affecting long-running processes.
- **Serialization for HttpUrl and datetime**: Fixed an issue where Pydantic v2's model_dump required mode="json" to handle non-primitive types correctly for program serialization.
- **Litellm Logging**: Default LiteLLM logging set to error, with explicit global logging configuration for debugging.
- **Test Coverage/Validation for Type and Code**: Added unit tests for dspy.Code/integration and type annotation extraction for more robust schema parsing and usage in Signatures.

## 2025-08-08 (v3.0.1)
### New Features
Welcome to 3.0.1!  This is a smaller fast follow release focused on enhancements of existing features, but we've still managed to sneak in some highly requested new features.  Biggest of them is that we've imported the GEPA package, woohoo!  We've also added type hints for the UsageTracker module, to make it easier to use and minimize upfront errors.  We hope you enjoy both of these and how much easier they make using the framework.

### Enhancements
The biggest enhancement here is removing the hardcoded model list from the OpenAIProvider file so that it will now dynamically update and provide a fully up to date list to he user qhen queried.  This enables it to be used in automated operations and prevents being stuck in legacy mode.  We've also made the handling of tool calls more flexible, so that the package can intelligently choose which ones are most appropriate. 

### Bug Fixes
A couple small but mighty bug fixes - the Evaluate call bug in GEPA is resolved and also when a reasoning model's requirements are not met we raise an explicit error instead of just silently faililng.

## 2025-08-01 (v3.0.0)
### New Features
We have a really exciting set of new features for our flagship version 3.0.0 release today!  First and foremost it's the addition of GPT-5 as a new reasoning model option.  This lets users leverage the latest and greatest LLM that OpenAI has provided.  Alongside that, we've incorporated LiteLLM for logging management and even built a custom BAMLAdapter to improve structured outputs.  Not it will be easier than ever before to ensure your LLM outputs are structured in a way that can be consumed and evaluated by machine learning models.

### Enhancements
The most important enhancements to existing features is in our JSON handling, both using JSON mode for serialization and also speeding up the JSON parsing for images.  On top of that, we now explicitly mark the end of streaming so that it is easier to programmatically identify through API commands. 

### Bug Fixes
Two critical fixes are included in this release, but if you're interested then check out the full changelog to get a comprehensive list.  The biggest is that we've fixed incorrect build requirements!  We made a mistake in the previous release of removing certain dependencies from the installation sequence even though they were still required in order for DSPy to function - no longer!  Now your installation will actually work :)  We've also fixed MCP tool conversion so that it works even when no input pararmeters are provided with the schema. 
