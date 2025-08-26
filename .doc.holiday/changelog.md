## 2025-07-23
### New Features
- **BAMLAdapter**: Custom adapter for highly readable nested JSON output with pydantic support and BAML schema-style rendering.
- **Code Type Support**: Added `dspy.Code` as a structured type for code generation and code analysis tasks, supporting language specifiers like `dspy.Code["python"]`.
- **History/Trace Buffer Size Control**: New settings `max_history_size` and `max_trace_size` allow control over history/trace buffer lengths for modules and LM objects.
- **Sampling and Hashing Improvements**: Swapped to in-package efficient Hasher to avoid dependency on datasets.

### Enhancements
- **Deduplicate Utility**: Improved token-efficient deduplication in utils by using dict.fromkeys.
- **Custom Type Parsing**: JSONAdapter and BAMLAdapter now more robustly handle custom and nested types, including fallback parsing and clearer error propagation on schema validation errors.
- **Cache Key Serialization**: Improved cache key computation to minimize unnecessary serialization and avoid side effects on original requests.
- **LiteLLM Logging Controls**: Revised LiteLLM logging switches for a cleaner user experience.
- **Pydantic Model Dumping**: Adopted `model_dump(mode="json")` in Predict serialization for better compatibility with pydantic's JSON/URL field types.
- **Warning on Direct Module.forward Use**: Now warns if users mistakenly invoke `.forward()` rather than the recommended instance-style `module()` call.
- **Toolkit/ToolCalls Expansion**: Expanded Tool and ToolCalls handling for complex system chaining and output rendering.

### Bug Fixes
- **Default Value Collision**: Resolved bug with default values in tool schemas when arg-desc was provided and values were missing.
- **List of Pydantic Models in Code Adapter**: Fixed rendering for lists of pydantic models by properly indenting and bracketing schemas.
- **Circular Reference Protection**: Simplified schema builder now guards against recursion and prints a placeholder for circular/forward refs.
- **Async Streaming**: Corrected handling of the last chunk and chunk spacing in streaming listeners to fix spacing glitches in streamed responses.
- **Avoid Hashing Unhashable Items in Proposal Code**: Improved robustness in get_dspy_source_code proposal util to avoid crashing on unhashable objects like module histories.
- **Test Suite Warnings**: Marked async tests involving warning messages to prevent pytest logs from showing unnecessary warnings.
- **Doc String and Typo Fixes**: Cleaned up doc typos and improved consistency in explanations and user-facing docstrings throughout tutorials, adapters, and utility functions.