# Remote Colab Embeddings Integration Guide

## 📋 Overview

This guide shows how to integrate **RemoteColabEmbeddings** into your RAG pipeline, replacing local embedding models with a remote Colab instance via ngrok.

### Why Use Remote Embeddings?
- 🚀 **No GPU/VRAM needed** - All embedding computation happens on Colab
- 💾 **Smaller app footprint** - Skip downloading multi-GB embedding models
- 🔄 **Flexible model updates** - Change embedding model without redeploying app
- 🌐 **Scalable** - Share single Colab instance across multiple services
- 💰 **Cost-effective** - Free Colab instead of expensive GPU instances

---

## 🔧 Setup

### Step 1: Set Your Colab URL

When your Colab embedding server starts, it generates an ngrok URL. Update this before running:

**PowerShell (Windows):**
```powershell
$env:COLAB_API_URL = "https://your-ngrok-url"
```

**Bash (Linux/Mac):**
```bash
export COLAB_API_URL="https://your-ngrok-url"
```

**Or create `.env` file in project root:**
```
COLAB_API_URL=https://your-ngrok-url
```

### Step 2: Install Dependencies

The `RemoteColabEmbeddings` class only requires `requests` and `langchain_core`:

```bash
pip install requests langchain-core
```

For vector database examples:
```bash
pip install chromadb langchain-community
```

---

## 🚀 Quick Start

### Simplest Usage

```python
from remote_embeddings import RemoteColabEmbeddings

# Initialize with your Colab URL
embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

# Get embedding for a single text
embedding = embeddings.embed_query("Patient has chest pain")

# Get embeddings for multiple texts
embeddings_list = embeddings.embed_documents([
    "High blood pressure symptoms",
    "Diabetes management tips",
    "Heart disease prevention"
])
```

---

## 📚 Integration Examples

### Example 1: With ChromaDB

```python
import chromadb
from remote_embeddings import RemoteColabEmbeddings

# Initialize embeddings
embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("medical_knowledge")

# Sample documents
documents = [
    "Hypertension is elevated blood pressure",
    "Diabetes increases cardiovascular risk",
    "Regular exercise prevents heart disease"
]

# Embed documents
doc_embeddings = embeddings.embed_documents(documents)

# Add to ChromaDB
collection.add(
    documents=documents,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Query the collection
query_embedding = embeddings.embed_query("What causes high blood pressure?")
results = collection.query(query_embeddings=[query_embedding], n_results=3)

print("Found documents:", results['documents'][0])
```

### Example 2: With LangChain Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from remote_embeddings import RemoteColabEmbeddings

# Initialize embeddings
embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

# Create documents
documents = [
    Document(
        page_content="Myocardial infarction occurs when blood flow to heart is blocked",
        metadata={"topic": "cardiac"}
    ),
    Document(
        page_content="Atrial fibrillation causes irregular heartbeat",
        metadata={"topic": "cardiac"}
    ),
]

# Create vector store with remote embeddings
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="medical_knowledge"
)

# Search
results = vector_store.similarity_search("What is a heart attack?", k=2)
for result in results:
    print(result.page_content)
```

### Example 3: With LangChain RAG Pipeline

```python
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from remote_embeddings import RemoteColabEmbeddings

# Initialize embeddings
embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

# Load vector store
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="medical_knowledge"
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask question
response = qa_chain("What causes heart disease?")
print(response["result"])
```

---

## 🔄 Using Your Existing Vector Store

If you already have a ChromaDB or other vector store, you can load it with remote embeddings:

### Replace Local Embeddings

**Before (local embeddings):**
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

**After (remote embeddings):**
```python
from remote_embeddings import RemoteColabEmbeddings

embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

---

## 🛡️ Error Handling

The `RemoteColabEmbeddings` class includes automatic retry logic and error handling:

```python
from remote_embeddings import RemoteColabEmbeddings

try:
    embeddings = RemoteColabEmbeddings(
        base_url="https://your-ngrok-url",
        timeout=30,           # Request timeout in seconds
        max_retries=3,        # Retry attempts on failure
        retry_delay=1.0       # Delay between retries
    )
    
    embedding = embeddings.embed_query("Medical text")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

### Fallback Strategy

Implement a fallback to local embeddings if Colab is unavailable:

```python
from remote_embeddings import RemoteColabEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    api_url = os.getenv("COLAB_API_URL")
    
    if api_url:
        try:
            return RemoteColabEmbeddings(base_url=api_url, timeout=5)
        except Exception as e:
            print(f"Colab unavailable, falling back to local: {e}")
    
    # Fallback to local embeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()
```

---

## 🔍 Monitoring & Debugging

### Check Service Status

```python
from remote_embeddings import RemoteColabEmbeddings

embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")

# Check if service is healthy
try:
    test_embedding = embeddings.embed_query("test")
    print("✅ Service is healthy")
except Exception as e:
    print(f"❌ Service error: {e}")
```

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("remote_embeddings")

# Now you'll see all debug messages
embeddings = RemoteColabEmbeddings(base_url="https://your-ngrok-url")
```

### Update URL If Colab Restarts

If your Colab restarts and generates a new ngrok URL:

```python
embeddings = RemoteColabEmbeddings(base_url="old-url")

# Later, when you get new URL
embeddings.update_base_url("https://new-ngrok-url")
```

---

## 📊 Performance Notes

### Throughput
- Single embedding: ~0.5-2 seconds depending on text length
- Batch of 100 texts: ~1-5 minutes
- Consider caching results for frequently searched texts

### Optimization Tips

1. **Batch operations**: Embed multiple texts together
   ```python
   # Good - single request per 100 texts
   embeddings.embed_documents(["text1", "text2", ..., "text100"])
   
   # Bad - 100 separate requests
   for text in texts:
       embeddings.embed_query(text)
   ```

2. **Cache results**: Store embeddings to avoid re-computing
   ```python
   import json
   
   cache = {}
   text_hash = hash(text)
   if text_hash in cache:
       embedding = cache[text_hash]
   else:
       embedding = embeddings.embed_query(text)
       cache[text_hash] = embedding
   ```

3. **Adjust timeout**: Longer timeout for larger texts
   ```python
   embeddings = RemoteColabEmbeddings(
       base_url="...",
       timeout=60  # More time for large batches
   )
   ```

---

## 🚨 Troubleshooting

### Connection Refused
**Problem:** `ConnectionError: Connection refused`
**Solution:** 
- Check ngrok URL is correct
- Verify Colab server is running
- Ensure ngrok tunnel is still active

### Timeout Errors
**Problem:** `Timeout after 30s`
**Solution:**
- Increase timeout: `RemoteColabEmbeddings(..., timeout=60)`
- Reduce batch size (embed fewer texts at once)
- Check Colab server CPU/memory usage

### API URL Not Set
**Problem:** `ValueError: base_url cannot be empty`
**Solution:**
- Set environment variable: `$env:COLAB_API_URL = "..."`
- Pass URL directly: `RemoteColabEmbeddings(base_url="...")`
- Add to `.env` file and load with `python-dotenv`

### Invalid Response Format
**Problem:** `Invalid response format: {}`
**Solution:**
- Check Colab server is serving correct API format
- Verify response includes `{"embedding": [...]}`
- Check server logs for errors

---

## 📝 File References

### Files Created
1. **`remote_embeddings.py`** - Main LangChain integration class
2. **`rag_integration_example.py`** - Complete working examples
3. **`embedding_colab.py`** - Lower-level service class (if not using LangChain)

### Usage in Your App

Modify your main app initialization:

```python
# In main.py or your RAG initialization
from remote_embeddings import RemoteColabEmbeddings
import os

# Initialize embeddings
api_url = os.getenv("COLAB_API_URL")
if api_url:
    embeddings = RemoteColabEmbeddings(base_url=api_url)
    print("✅ Using remote Colab embeddings")
else:
    # Fallback or error
    print("⚠️  COLAB_API_URL not configured")

# Pass to your vector store
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

---

## ✅ Testing

### Run the Integration Examples

```bash
# Set environment variable
$env:COLAB_API_URL = "https://your-ngrok-url"

# Run examples
python rag_integration_example.py
```

### Expected Output
```
✅ Example 1: Basic Text Embedding
📝 Embedding: What causes coronary artery disease?
✅ Got embedding (dimension: 384)

✅ Example 2: ChromaDB Integration
📚 Adding 5 documents to ChromaDB...
✅ Documents added to ChromaDB
🔍 Testing similarity search...
✅ Found 3 relevant documents:
   [1] Hypertension is elevated blood pressure above 130/80 mmHg (distance: 0.1234)
```

---

## 📞 Support & Next Steps

### Common Scenarios

**Scenario 1: Adding new documents after app startup**
```python
# Get existing vector store with remote embeddings
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Add new documents
new_docs = [Document(page_content="...")]
vector_store.add_documents(new_docs)  # Automatically uses embeddings
```

**Scenario 2: Switching between embeddings**
```python
# During development - use local embeddings
dev_embeddings = HuggingFaceEmbeddings()

# In production - use remote embeddings
prod_embeddings = RemoteColabEmbeddings(base_url=api_url)

embeddings = prod_embeddings if production else dev_embeddings
```

**Scenario 3: Multi-user setup**
```python
# One Colab server can serve multiple services
service1_embeddings = RemoteColabEmbeddings(base_url=shared_colab_url)
service2_embeddings = RemoteColabEmbeddings(base_url=shared_colab_url)
# Both connect to same Colab instance
```

---

## 🎯 Summary

**Key Points:**
- ✅ Drop-in replacement for local embedding models
- ✅ Works with ChromaDB, LangChain, and other tools
- ✅ Automatic retry and error handling
- ✅ Configurable timeout and batch processing
- ✅ Perfect for resource-constrained environments

**Next Steps:**
1. Set `COLAB_API_URL` environment variable
2. Import `RemoteColabEmbeddings` in your app
3. Pass to ChromaDB or LangChain vector store
4. Test with provided examples
5. Deploy with confidence! 🚀
