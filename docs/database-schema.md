# MRtrix3 RAG Database Schema

## Overview
The MRtrix3 Agent uses a Supabase-hosted PostgreSQL database to store and search MRtrix3 documentation. Each documentation page is stored as a single unit with semantic embeddings for intelligent retrieval.

## Database Details
- **Platform**: Supabase PostgreSQL
- **Vector Embeddings**: 768-dimensional Google Gemini embeddings

## Documents Table Structure

### Core Fields
- **id** (`UUID`): Unique identifier for each document
- **title** (`TEXT`): Document title (e.g., "mrconvert.rst")
- **content** (`TEXT`): Full document content in markdown/plain text
- **content_embedding** (`VECTOR(768)`): Semantic embedding for similarity search
- **source_url** (`TEXT`): Original documentation URL for reference

### Classification Fields
- **doc_type** (`TEXT`): Document type - one of:
  - `command`: MRtrix3 command reference
  - `tutorial`: Step-by-step guides
  - `guide`: Conceptual documentation
  - `reference`: Technical API documentation
- **command_name** (`TEXT`): Command name (only for command docs, e.g., "tckgen")
- **version** (`TEXT`): MRtrix3 version (e.g., "3.0.4", "latest")

### Searchable Metadata
- **keywords** (`TEXT[]`): Array of searchable terms
- **concepts** (`TEXT[]`): MRI/diffusion concepts covered (e.g., ["b-value", "tensor"])
- **synopsis** (`TEXT`): Brief summary of the document

### Command-Specific Fields
These fields are only populated for `doc_type='command'`:
- **command_usage** (`JSONB`): Examples and parameters
  ```json
  {
    "command_list": ["mrconvert input.mif output.nii"],
    "parameter_list": ["-stride", "-force", "-grad"]
  }
  ```
- **error_types** (`JSONB`): Error messages by function
  ```json
  {
    "run": "axis supplied to option -axes is out of bounds",
    "validate_options": "Option -grad and -fslgrad are mutually exclusive"
  }
  ```

### Timestamps
- **created_at**: When document was added
- **updated_at**: Last modification time
- **last_updated**: When content was last refreshed
