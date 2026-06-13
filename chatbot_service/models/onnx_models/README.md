# ONNX Models (all-MiniLM-L6-v2)

## Why we use this model
This folder contains the quantized ONNX version of the `all-MiniLM-L6-v2` embedding model. We use this model to generate dense vector embeddings for our medical documents and user queries within the RAG pipeline. By utilizing the ONNX framework, the model executes much faster and with a significantly lower memory footprint compared to a standard PyTorch implementation. This high performance ensures rapid retrieval of relevant medical text chunks from our vector database while keeping computing resource usage highly efficient.
