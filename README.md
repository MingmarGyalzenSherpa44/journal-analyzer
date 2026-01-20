# Journal Analyzer RAG Project

A Retrieval Augmented Generation (RAG) application that analyzes personal journal entries using AWS Bedrock and LangChain.

## Features

- **Document Ingestion**: Load and process journal entries from text files
- **Vector Storage**: Use ChromaDB for efficient semantic search
- **AWS Bedrock Integration**: Leverage Claude 3.5 Sonnet and Titan embeddings
- **Interactive UI**: Streamlit-based chat interface
- **Source Attribution**: View relevant journal excerpts for each answer

## Setup

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- AWS credentials configured

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

4. Add journal entries:
```bash
# Place your .txt journal files in data/journals/
```

### Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

### Example Journal Entry Format

Create text files in `data/journals/`:

**2024-01-15.txt**:
```
Today was a great day. I finished the project I've been working on for weeks.
Feeling accomplished and ready for new challenges.
```

## Architecture

1. **Embeddings**: Amazon Titan Text Embeddings
2. **Vector Store**: ChromaDB for local storage
3. **LLM**: Claude 3.5 Sonnet via Bedrock
4. **Framework**: LangChain for orchestration

## Configuration

Edit `src/config.py` or `.env` to customize:
- Model selections
- Chunk sizes
- Number of retrieved documents
- AWS region

## Project Structure

```
journal-analyzer-rag/
├── src/
│   ├── config.py          # Configuration management
│   ├── embeddings.py      # Bedrock embeddings service
│   ├── vector_store.py    # ChromaDB vector store
│   ├── retriever.py       # Document retrieval
│   ├── llm.py            # Bedrock LLM service
│   └── rag_chain.py      # RAG chain orchestration
├── data/
│   ├── journals/         # Your journal entries (.txt)
│   └── vector_db/        # ChromaDB storage
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## License

MIT