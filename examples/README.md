# Examples

Run these commands from the repository root.

## Test mock

Use the test mock for a quick local check with no model or GPU. `demo.py` starts `tests/mock_server.py` in process, exactly as the tests do, and scores the sample items.

```console
python examples/demo.py
```

## Chat server

Use the chat backend for an OpenAI-compatible server that supports logprobs and `continue_final_message`.

```console
gavel score examples/reviews/rubric.yaml examples/reviews/items.jsonl --backend openai --base-url http://127.0.0.1:8000 --model quennox-judge
```

## Completions server

Use the completions backend when the engine does not support `continue_final_message`.

```console
gavel score examples/reviews/rubric.yaml examples/reviews/items.jsonl --backend completions --prompt-format chatml --base-url http://127.0.0.1:8000 --model quennox-judge
```

## Native model

Use the native backend for a full-attention Hugging Face model loaded in process.

```console
gavel score examples/reviews/rubric.yaml examples/reviews/items.jsonl --backend native --model-path "$MODEL_DIR" --device cuda --dtype bfloat16
```
