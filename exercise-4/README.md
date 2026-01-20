This module implements a Retrieval-Augmented Generation (RAG) workflow using Azure OpenAI and ChromaDB.  
The system retrieves relevant documents from the vector store and passes them as grounded context to an LLM, which answers user questions using only the provided context.  
If the required information is not present in the dataset, the model responds with “I don’t know,” ensuring reliable and hallucination-free behavior.  
The module also shows the source documents for each answer to provide transparency and traceability.