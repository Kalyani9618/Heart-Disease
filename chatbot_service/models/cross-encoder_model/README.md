# Cross-Encoder Model

## Why we use this model
In our project, the cross-encoder model is used as a reranker for our Retrieval-Augmented Generation (RAG) pipeline. While standard embedding models retrieve a broad set of relevant chunks quickly, the cross-encoder precisely scores the relevance of the retrieved document chunks against the user's specific query. By reranking the returned results, we ensure that only the most highly relevant and contextually accurate medical information is passed to the LLM. This significantly improves the accuracy and trustworthiness of the text generation.
