# FYI; For Your Information

This project enables intuitive searching of local files and documentation using natural language queries. Powered by a Large Language Model (LLM), it interprets human-written queries and locates relevant files and content efficiently. Under the hood, a vector store is used to enhance search accuracy and performance.

## Features

- **Natural Language Search:** Find files and documentation using queries written in plain English.
- **LLM-Powered Interpretation:** Queries are interpreted by an LLM for better understanding and relevance.
- **Vector Store Backend:** Fast and accurate search results using vector embeddings.
- **Local File Support:** Works with files stored on your local machine.

## Getting Started

1. Clone this repository.
2. Install dependencies.
3. Run the application and start searching your local files with natural language queries.

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python main.py
> Ask a question: what is a rapidly evolving field in my case?
========================================
[ 1.49 secs ]

Query: what is a rapidly evolving field in my case?
Answer: Deep Learning
Sources:
* .../FYI/target/sample4.html (1)
* .../FYI/target/sample5.html (1)
* .../FYI/target/sample3.html (1)
* .../FYI/target/sample2.html (1) 
```

## Setting Environment Variables

Before running the application, export the required environment variables in your terminal:

```bash
export HF_HOME=~/.cache/huggingface
export HF_AUTH_TOKEN=your_api_key_here
```

Replace the values with your actual API key and vector store path.

## Usage

Simply type your query (e.g., "Show me the documentation for the authentication module") and the system will locate the relevant files and content.

## License

MIT

## Contributing

Contributions are welcome! Please open issues or submit pull