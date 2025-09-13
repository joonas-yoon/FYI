# FYI; For Your Information

This project enables intuitive searching of local files and documentation using natural language queries. Powered by a Large Language Model (LLM), it interprets human-written queries and locates relevant files and content efficiently. Under the hood, a vector store is used to enhance search accuracy and performance.

## Features

- **Natural Language Search:** Find files and documentation using queries written in plain English.
- **LLM-Powered Interpretation:** Queries are interpreted by an LLM for better understanding and relevance.
- **Vector Store Backend:** Fast and accurate search results using vector embeddings.
- **Local File Support:** Works with files stored on your local machine.

## Prerequisites

This project requires both Ollama and LangChain:

- **Ollama and LangChain:** Install and set up [Ollama](https://python.langchain.com/docs/integrations/llms/ollama/#setup) and ensure [LangChain](https://python.langchain.com/docs/) is installed as a Python dependency.

Refer to the official documentation for instructions on configuring Ollama with LangChain.

## Get Started

1. Clone this repository.
2. Install dependencies.
3. Run the application and start searching your local files with natural language queries.

## Usage and Example

Simply type a question in the terminal, and the system will search your local files and documentation, returning relevant answers with source references.

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

## License

MIT

## Contributing

Contributions are welcome! Please open issues or submit pull