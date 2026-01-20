This module converts natural language music requests into structured JSON filters that can be applied directly to ChromaDB metadata.  
The LLM interprets user intent (e.g., "acoustic songs under 2 minutes, not explicit") and outputs a strict, validated JSON object following a predefined schema.  
The filters can express conditions such as duration ranges, genre restrictions, popularity thresholds, and explicit flags.  
This component is a core part of the upcoming Playlist Assistant, enabling flexible semantic + metadata-based querying.