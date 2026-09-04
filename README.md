# Agentic Research Assistant: Edge AI & Gesture Tracking Literature Triage

**Live Streamlit Demo:** [Insert your Streamlit Community Cloud URL here][cite: 1]

## 1. Project Description
This Agentic AI application is a research support tool designed to automate literature triage, methodology extraction, and structural analysis of academic datasets. It directly supports ongoing research in lightweight edge-based computer vision, specifically focusing on optimizing MobileNet and YOLO architectures via TensorRT, as well as analyzing MediaPipe Holistic landmark extraction techniques for custom sign language datasets. 

## 2. Architecture & Agentic Design Patterns
This system implements three distinct agentic design patterns to handle complex queries:
1. **Router Pattern:** A lightweight routing function classifies the user's intent to determine if the query requires querying the vector database or if it is a general conversational request[cite: 1].
2. **Tool-Use (ReAct) Pattern:** The Extraction Agent is equipped with a retrieval tool, allowing it to search the vector database for specific algorithm implementations, coordinate normalization methods, and performance benchmarks[cite: 1].
3. **Reflection/Self-Critique Pattern:** A dedicated evaluation node reviews the Synthesis Agent's final output against the retrieved context to correct hallucinations or missing citations before delivering the response[cite: 1].

*(Insert Architecture Diagram Image Here)*[cite: 1]

## 3. Agent-to-Agent Communication
The workflow is orchestrated using LangGraph, establishing a structured message-flow protocol between two primary agents[cite: 1]. 
* **Extraction Agent:** Queries the RAG tool, parses the returned document chunks, and structures the findings into a standard JSON format.
* **Synthesis Agent:** Consumes the structured JSON from the Extraction Agent, synthesizes a comparative literature response, and initiates the self-critique loop.

*(Insert Sequence/Message-Flow Diagram Image Here)*[cite: 1]

## 4. Model Selection Strategy
Models were deliberately selected across Groq and OpenRouter to optimize for cost, latency, context length, and deep reasoning[cite: 1].

| Sub-task | Model (Provider) | Why Chosen[cite: 1] |
| :--- | :--- | :--- |
| Intent Routing & Classification | Llama 3.1 8B (Groq) | Extremely low latency and near-zero cost; highly efficient for simple binary routing decisions without bottlenecking the pipeline. |
| Tool-Use & Data Extraction | Mixtral 8x7B (Groq) | Fast token generation to quickly parse through multiple retrieved document chunks and extract key methodologies. |
| Deep Reasoning & Final Synthesis | Claude 3.5 Sonnet (OpenRouter) | Superior reasoning capabilities and a large context window justify the higher cost/latency for complex academic synthesis and formatting. |

## 5. RAG Integration
* **Domain Corpus:** 20+ full-text academic PDFs covering TensorRT optimization, edge-based object detection, and three-dimensional landmark tracking methodologies[cite: 1].
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` utilizing 1000-character chunks with a 200-character overlap to preserve sentence boundaries and mathematical contexts[cite: 1].
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` for generating dense vector representations[cite: 1].
* **Vector Store:** Local ChromaDB deployment[cite: 1].
* **Retrieval Evaluation:** The pipeline was tested against 5 sample queries (e.g., "Extract normalization techniques for shoulder center coordinates"). The retrieved context accurately surfaced the relevant methodologies with zero critical hallucinations[cite: 1].

## 6. Setup Instructions & Deployment
1. Clone the repository: `git clone https://github.com/yourusername/your-repo-name.git`
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Create a `.streamlit/secrets.toml` file in the root directory and add your API keys:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   OPENROUTER_API_KEY = "your_openrouter_key"
