# MRtrix3 Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-0.0.14-green.svg)](https://ai.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## AI assistant for MRtrix3 documentation and workflow guidance in the terminal

https://github.com/user-attachments/assets/0230bb50-9a90-453a-9c52-f0da3e75193a

## How does the MRtrix3-agent work?

This project serves as a simple implemention of Agentic AI knowledge base generation and retrieval. This foundation can be built on to implement custom workflows or transfer the framework to other documented software. The general structure of this repo is as follows:
1. Official MRtrix3 documentation (latest version) is stored in a PostgreSQL database (Supabase)
2. Users install a chromaDB locally which fetches documents from Supabase and stores them locally
   - Upon every app startup, chromaDB checks itself against Supabase (ground truth) to determine if sync for up-to-date documentation is needed
3. User chats with AI Agent (Gemini-2.5-flash)
4. To provide accurate answers, AI Agent is equipped with a search tool to reference relevant and up-to-date documentation
5. Monitoring and log collection is available and updated in real-time to view Agent queries and retrievals in real-time

From this basic structure, additional tools or extensions to the AI Agent's capabilities can be easily added. As an example, the current version of the repo comes with 3 sample slash commands, one of which ("/sharefile") allows the user to share their file metadata with the Agent using MRtrix's 'mrinfo' for personalized guidance.

## 📋 Prerequisites

- Python 3.10 or higher
- API keys for your Gemini (Google)
  - The code currently only supports Gemini as Google offers a generous free tier for API calls.
  - Google API key can be generated from Google AI Studio, docs here: https://ai.google.dev/gemini-api/docs/api-key
- Currently only tested on Linux systems

## For Regular Use

1. Install with pip
```bash
pip install mrtrix3-agent
```

2. Setup your API key (you will be prompted to enter)
```bash
setup-mrtrixbot
```

3. Run the agent and chat!
```bash
mrtrixbot
```

## 🛠️ For Contributors/Developers

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mrtrix3-agent.git
cd mrtrix3-agent
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e . # For standard usage
# or
pip install -e ".[dev]" # If you want to run test scripts with pytest
```

### 4. Configure Environment Variables

'.env.example' contains configurable environment variables used in this project. The project was built with Supabase and Gemini. If
wanting to replicate this project, a Supabase account is required and all the tools to populate your own database can be found in the
src/knowledge_base folder. Great for anyone curious about how a knowledge base gets created, or for those trying to maybe create a
"smarter" knowledge base for the AI agent!

### 5. Initialize Database (Optional to replicate project)

Populate the Supabase vector database with MRtrix3 documentation:

```bash
python knowledgeBase/populate_database.py
```

This will:
- Clone MRtrix3 repository (sparse checkout)
- Process documentation and source files
- Generate embeddings and populate the database
- Takes approximately 1-5 minutes

## 🎯 Usage

### Basic Usage

Start the interactive CLI:

```bash
mrtrix3-agent
```

Or use the Python script directly:

```bash
python src/agent/cli.py
# or
python -m src.agent.cli
```
## 📊 Monitoring

The agent includes built-in monitoring:

- Set environment variable 'COLLECT_LOGS=true'
- Session logs in `monitoring/logs/`
- Performance metrics in `monitoring/metrics/`
- Error tracking with detailed stack traces

Logs within `monitoring/logs/` get updated in real-time, good to look at if you are curious where the AI agent is searching for info
to answer queries!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

You can reach me at rmoskwa@wisc.edu
---
