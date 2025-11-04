# Generative Support Agent (Step 7)

This module exposes a lightweight FastAPI endpoint that answers natural-language questions about the FusionGuard project. It uses retrieval over the existing documentation and can optionally call a local `llama.cpp` model for richer language generation. When an LLM is not available the service falls back to a deterministic, rules-based response so that the API remains functional in completely offline environments.

## Key Components
- `src/agent/vector_store.py` – builds a FAISS-backed semantic index (with automatic TF–IDF fallback) over the Markdown files in `docs/` to retrieve relevant context snippets.
- `src/agent/prompt_manager.py` – crafts system prompts with safety guardrails, persona-specific guidance, and the retrieved context.
- `src/agent/llm_client.py` – abstract interface for LLM backends. Includes:
  - `LlamaCppClient` (requires `llama-cpp-python` and a local GGUF model path passed via `LLAMA_CPP_MODEL`).
  - `FallbackLLMClient` for deterministic replies when no generative model is available.
- `src/agent/service.py` – FastAPI app exposing `/healthz` and `/v1/assist` endpoints.

## Running the API Locally
```bash
# Optional: activate your project virtualenv
source .venv/bin/activate

# Optionally point to a llama.cpp model (GGUF file)
export LLAMA_CPP_MODEL=/path/to/llama-7b.Q4_K_M.gguf

# Launch the service
uvicorn src.agent.service:app --reload --port 8005
```

Request example:
```bash
curl -X POST http://localhost:8005/v1/assist \
  -H "Content-Type: application/json" \
  -d '{"query": "How should I explain the churn model performance to customer success?", "persona": "support"}'
```

Response payload:
```json
{
  "response": "### Summary\n ...",
  "sources": [
    {"source": "docs/churn_model_card.md", "score": "0.631", "preview": "…"}
  ]
}
```

## Persona Support
Passing the `persona` field adjusts tone and emphasis. Available personas:

| Persona   | Notes                                                                 |
|-----------|-----------------------------------------------------------------------|
| support   | Empathetic tone, suggests escalation paths for customers.             |
| executive | Focus on business impact, concise bullet-style answers.               |
| analyst   | Highlights quantitative metrics and analytic follow-up actions.       |

## Retrieval Architecture
- Sentence embeddings powered by `sentence-transformers/all-MiniLM-L6-v2` feed into a FAISS inner-product index for low-latency, semantic retrieval across project documentation.
- If FAISS or the embedding model cannot be initialised (e.g., strictly offline environments without the model cached), the service automatically falls back to the legacy TF–IDF strategy to remain operational.
- To pre-cache embeddings for offline clusters, download the model once on a network-enabled box:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
  ```
  Copy the resulting Hugging Face cache directory (default `~/.cache/huggingface`) onto the air-gapped HPC environment before launching the API.

## Production Checklist
- Install `llama-cpp-python` and download a GGUF model if generative responses are required.
- Pre-load the sentence-transformer embeddings cache (or confirm FAISS initialises) on the target environment.
- Add authentication (e.g., API key) before exposing the endpoint publicly.
- Extend monitoring by logging prompt/response payloads with redaction for sensitive fields.

## HPC Deployment Notes
Interruptible GPU nodes on CREATE run NVIDIA GeForce RTX 2080 Ti hardware (compute capability `sm_75`). Builds of `llama-cpp-python` that only include newer architectures (`sm_80`, etc.) will crash with `CUDA error: no kernel image is available for execution on the device`. Rebuild the wheel on a GPU node with the matching architecture and ensure the CUDA libraries are on the search path:

```bash
module purge
module load gcc/11.4.0
module load cuda/12.2.1
source ~/fusionguard-analytics/.venv_hpc/bin/activate

export CUDA_ROOT=/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/cuda-12.2.1-rqt4vr3vbmq5edgzd5glgq73htbneaw6
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib:/usr/lib/gcc/x86_64-linux-gnu/11:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:${LIBRARY_PATH:-}

pip install llama-cpp-python==0.3.3 \
  --config-settings=cmake.define.GGML_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=75 \
  --config-settings=cmake.define.LLAMA_BUILD_EXAMPLES=OFF \
  --config-settings=cmake.define.LLAMA_BUILD_TESTS=OFF \
  --config-settings=cmake.define.LLAMA_BUILD_SERVER=OFF \
  --config-settings=cmake.define.LLAMA_BUILD_LLAMAFILE=OFF \
  --config-settings=cmake.define.LLAMA_BUILD_LLAVA=OFF \
  --no-cache-dir
```

After rebuilding, run a quick import test on the GPU node to confirm the model loads:

```bash
python - <<'PY'
from llama_cpp import Llama
Llama(
    model_path="/scratch/users/<user>/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4,
)
print("loaded ok")
PY
```

Only once this succeeds should you launch the batch job (`sbatch llama_gpu.sh`). This avoids the `RMS_NORM failed` / `no kernel image` crashes seen with mismatched CUDA architectures.
