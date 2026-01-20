This module loads a CSV dataset of Spotify tracks, transforms each row into a normalized document, and generates embeddings using Azure OpenAI.  
The script batches the embedding requests, handles rate limits with retry logic, and stores the vectors in a persistent ChromaDB index.  
Each document is assigned a unique ID to avoid duplicates, and a checkpoint file is used to support resumable ingestion.  
After running the script, a complete local vector database is created and ready for semantic search and RetrievalQA.