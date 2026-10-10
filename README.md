# vLLM Asychronous Optimization 
![vLLM](https://img.shields.io/badge/vLLM-GPU_Inference-FF6F00?style=flat)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey) 

This code leverages vLLM and the OpenAI API wrapper for faster, asynchronous LLM inference using coroutines. It was developed over May 2025-May 2026 via extensive experiments over high-performance NVIDIA GPUs over university servers.

The implementation leverages available GPU resources for local server endpoint hosting via the OpenAI API completion wrapper and vLLM's fast inference capability.
The code offers a walkthrough of resource optimization methods, including optional parallel compute and multi-server hosting. 

The idea is inspired by coroutines and async background tasks from prior experience. Proudly developed.

# `vllm_async_vr.py`

A small asynchronous smoke-test/demo client for a vLLM server exposing the OpenAI-compatible Chat Completions API. It submits country-capital questions to a served chat model and prints one generated answer.

> This script does **not** calculate accuracy or compare model outputs with the `capital` values in its sample data. It is not a benchmark or a reproducible evaluation harness as written.

## What the script does

1. Starts `evaluate()` with your dataset records.
2. Builds a system message (`You are a helpful assistant.`) and a user question for each country.
3. Divides the prompts into 30 NumPy chunks and processes the chunks in order. Prompts within each chunk are submitted concurrently with `asyncio.gather()`.
4. Sends chat-completion requests to `http://127.0.0.1:8000/v1`, using the model name of your choice e.g., `meta-llama/Llama-3.1-8B-Instruct` and a 24-token generation limit by default.
5. Lowercases each returned message and prints only the first response as `Example response: ...`.

Generation settings not supplied by the client, such as temperature, are left to the server's defaults. The script does not save prompts, responses, or run metadata.

## Requirements

- Python with the packages `openai`, `aiohttp`, `numpy`, and `tenacity` installed.
- A reachable vLLM OpenAI-compatible server with the requested chat model loaded.
- GPU/CUDA requirements depend on the model and the vLLM server environment; this client itself makes HTTP requests and does not load the model onto a GPU.

Install the client-side Python packages into the environment used to run the script:

```bash
python -m pip install openai aiohttp numpy tenacity
```

Install and configure vLLM separately in the server environment, following the installation guidance for the vLLM version and hardware in use.

## Run

From the workspace root, start the server in one terminal. For example, to serve the default model on the host and port expected by the script:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

Wait until vLLM reports that the server is ready. In another terminal, run the client:

```bash
python code/vllm_async_vr.py
```

A successful run prints one generated answer, for example:

```text
Example response: paris
```

The exact response is model-dependent. To capture output in a log:

```bash
python code/vllm_async_vr.py > eval.log 2>&1
```

The script has no command-line options. Its model name, server URL, prompts, and token limit must be changed in the source to use different values.

## Configuration points

The active server URL is hard-coded in `create_completion_prompt()` as `http://127.0.0.1:8000/v1`. Change it there if the server uses another port or host. Binding the server to `127.0.0.1` is appropriate when the client runs on the same machine; for a client on another machine, configure a reachable server address and an appropriately secured network path.

The active defaults are:

- Model: `meta-llama/Llama-3.1-8B-Instruct` (`evaluate()`'s `model_id` default)
- Maximum generated tokens: `24` (`evaluate()`'s `max_tokens` default)
- Sample prompts: eight country-capital questions in `main()`
- Response format: plain text

`main()` calls `evaluate(data)` without overriding these defaults. The `capital` field in each sample record is currently unused.

## Concurrency and retries

The script creates all tasks for one NumPy chunk and awaits them together. The semaphore is created inside each request function, so it does not limit concurrency across requests. `numpy.array_split(..., 30)` always creates 30 chunks: with the current eight-item sample, eight chunks contain one prompt each and 22 are empty. For larger inputs, NumPy divides the prompts into 30 approximately even chunks, and the chunk sizes are not a maximum-concurrency setting. Do not treat this chunking as a server capacity limit.

The request function uses Tenacity with randomized exponential backoff (1 to 10 seconds) and up to five attempts. It also catches OpenAI connection and rate-limit errors and calls `time.sleep(2)` before raising; because this is synchronous sleep inside an async function, it blocks the event loop during that delay. Connection failures commonly indicate that the server is not ready or that the hard-coded host/port does not match the server.

The OpenAI client is initialized with the placeholder API key `fake_key`, which is suitable only when the local vLLM server does not enforce API-key authentication. Do not use this configuration as authentication for a network-exposed service.

## Limitations to account for

- Only the first generated answer is printed; the other responses are not displayed or persisted.
- No answer normalization beyond lowercasing, gold comparison, metric calculation, or random seed is implemented.
- The `aiohttp.ClientSession` passed to the request function is not used for the API call; requests are made through `openai.AsyncOpenAI`.
- The request function creates a new OpenAI client for each prompt and does not explicitly close it.
- A missing or unusable response can be converted to the integer `0`, despite the function's `str` return annotation.
- The host and port are fixed in source, and this script has no built-in support for selecting among multiple servers.

For research evaluation, extend the script to accept an explicit dataset and server/model configuration, retain every prompt and response, compare outputs against a defined reference with a documented scoring procedure, and record versions, decoding settings, and run metadata. Validate the resulting harness before interpreting its outputs as experimental measurements.
