# MRtrix3 Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-0.0.14-green.svg)](https://ai.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Prerequisites

- Python 3.10 or higher
- API keys for your Gemini (Google)
  - The code currently only supports Gemini as Google offers a generous free tier for API calls.
  - Google API key can be generated from Google AI Studio, docs here: https://ai.google.dev/gemini-api/docs/api-key
- Currently only tested on Linux systems

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
