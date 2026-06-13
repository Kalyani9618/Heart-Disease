# MedGemma 4B Model

**Why**: MedGemma is a medical-specialized language model fine-tuned on clinical data, providing accurate and safe health-related responses for our HeartGuard AI assistant.

**How**: The model runs locally via LM Studio on port 8090, serving as the primary LLM for medical question answering, RAG-based retrieval, symptom analysis, and patient interaction through our FastAPI backend.

**Files**: `medgemma-4b-it-Q5_K_M.gguf` (main model) and `mmproj-medgemma-4b-it-F16.gguf` (vision projector for medical image analysis).
