#!/usr/bin/env python3
"""
MRtrix3 Documentation RAG Database Populator
Using sparse git checkout for efficient documentation gathering.

Usage:
    python -m knowledgeBase.populate_database [options]

Options:
    --batch-size N        Process only N documents (useful for testing)
    --include-existing    Reprocess documents already in database

Examples:
    # Process all new documents (default)
    python -m knowledgeBase.populate_database

    # Test with small batch
    python -m knowledgeBase.populate_database --batch-size 5

    # Reprocess everything (updates existing)
    python -m knowledgeBase.populate_database --include-existing

Features:
    - Sparse git checkout: Only downloads docs/ and source dirs (~5MB vs 200MB)
    - Concurrent processing: Processes up to 30 documents in parallel
    - Rate limiting: Token bucket algorithm respects Gemini 2.5 Flash limits (2000 RPM)
    - Gemini API: Analyzes RST documents for structured extraction
    - Error extraction: Finds error messages from C++/Python source code
    - Smart truncation: Uses strided sampling (30% start, 60% middle, 10% end)
    - Database validation: Ensures all fields meet Supabase constraints
    - Progress tracking: Rich terminal UI with progress bars
"""

import os
import re
import json
import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv
import logging

from supabase import create_client, Client
from google import generativeai
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MRTRIX3_REPO = "https://github.com/MRtrix3/mrtrix3.git"
GEMINI_CONCURRENT_LIMIT = 30  # Maximum concurrent Gemini API calls
GEMINI_REQUESTS_PER_SECOND = 30  # Rate limit: 30 req/s (1800 RPM, under 2000 limit)

# Initialize services
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
generativeai.configure(api_key=GEMINI_API_KEY)


class DocumentAnalysis(BaseModel):
    """Structure for Gemini's analysis of a document"""

    doc_type: str  # 'command', 'guide', 'tutorial', 'reference' (database constraint)
    command_name: Optional[str] = None
    concepts: List[str] = []
    synopsis: Optional[str] = None
    command_usage: Optional[str] = None


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, rate: float, per: float = 1.0):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default 1.0 for per-second)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to maintain rate limit"""
        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current

            # Replenish tokens based on time passed
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate

            # If not enough tokens, wait
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0


class RSTDocumentGatherer:
    """Gather RST documents using sparse git checkout"""

    def __init__(self, repo_url: str = MRTRIX3_REPO):
        self.repo_url = repo_url

    def gather_documents_and_source(
        self, tmpdir: str, branch: str = "master"
    ) -> tuple[List[Dict[str, Any]], str, Path]:
        """
        Use sparse checkout to efficiently gather documentation and source files.
        Returns (documents, version, repo_path)
        NOTE: tmpdir must be managed externally to keep files available
        """
        repo_path = Path(tmpdir) / "mrtrix3"

        console.print(f"[cyan]📥 Sparse cloning from {self.repo_url}...")

        try:
            # Initialize sparse checkout
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    "--depth",
                    "1",
                    "--branch",
                    branch,
                    self.repo_url,
                    str(repo_path),
                ],
                check=True,
                capture_output=True,
            )

            # Set sparse checkout for docs AND source code directories
            subprocess.run(
                [
                    "git",
                    "sparse-checkout",
                    "set",
                    "docs/",
                    "cpp/cmd/",  # C++ commands for error extraction
                    "python/mrtrix3/commands/",  # Python commands for error extraction
                ],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            console.print("[green]✅ Successfully fetched docs and source code")

            # Get the most recent release tag (not commit hash)
            try:
                # First, fetch all tags
                subprocess.run(
                    ["git", "fetch", "--tags"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

                # Get the most recent tag that looks like a version (e.g., 3.0.7)
                version_result = subprocess.run(
                    ["git", "tag", "-l", "--sort=-version:refname"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )

                if version_result.returncode == 0 and version_result.stdout.strip():
                    # Filter for semantic version tags (e.g., 3.0.7, not dev tags)
                    tags = version_result.stdout.strip().split("\n")
                    for tag in tags:
                        # Match tags like 3.0.7, 3.0.0, etc.
                        if re.match(r"^\d+\.\d+\.\d+$", tag.strip()):
                            version = tag.strip()
                            break
                    else:
                        # If no semantic version found, use the first tag
                        version = tags[0].strip() if tags else "latest"
                else:
                    version = "latest"

            except subprocess.CalledProcessError:
                # Fallback to "latest" if tags fetch fails
                version = "latest"

            # Gather all RST files
            docs_dir = repo_path / "docs"
            documents = []

            for rst_file in sorted(docs_dir.rglob("*.rst")):
                # Skip build artifacts
                if "_build" in str(rst_file):
                    continue

                with open(rst_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title from RST
                title = self.extract_rst_title(content)

                # Determine doc type from path and content
                doc_type = self.classify_doc_type(rst_file, content)

                # Extract command name if applicable
                command_name = None
                if "commands/" in str(rst_file):
                    command_name = rst_file.stem

                documents.append(
                    {
                        "path": str(rst_file.relative_to(docs_dir)),
                        "title": title,
                        "content": content,
                        "doc_type": doc_type,
                        "command_name": command_name,
                        "url": f"https://mrtrix.readthedocs.io/en/latest/{str(rst_file.relative_to(docs_dir)).replace('.rst', '.html')}",
                    }
                )

            console.print(f"[green]📚 Gathered {len(documents)} RST documents")

            # Return documents, version, AND repo path for error extraction
            return documents, version, repo_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            raise

    def extract_rst_title(self, content: str) -> str:
        """Extract title from RST content"""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # RST titles are underlined with =, -, ~, etc.
                if (
                    next_line
                    and all(c in '=-~^"*+' for c in next_line.strip())
                    and len(next_line.strip()) >= 3
                ):
                    title = line.strip()
                    # Remove RST formatting like *emphasis*
                    title = re.sub(r"\*([^*]+)\*", r"\1", title)
                    return title
        return "Untitled"

    def classify_doc_type(self, file_path: Path, content: str) -> str:
        """Classify document type based on path and content (database constraint: command, tutorial, guide, reference)"""
        path_str = str(file_path)

        if "commands/" in path_str:
            return "command"
        elif "reference/" in path_str:
            return "reference"
        elif "getting_started/" in path_str or "tutorial" in path_str:
            return "tutorial"
        elif "Synopsis" in content and "Usage" in content:
            return "command"
        elif "concepts/" in path_str or "installation/" in path_str:
            # Map concepts and installation to guide since database doesn't have those types
            return "guide"
        else:
            return "guide"


class GeminiRSTAnalyzer:
    """Analyze RST documents using Gemini"""

    def __init__(self):
        self.model = generativeai.GenerativeModel("gemini-2.5-flash")
        self.rate_limiter = RateLimiter(rate=GEMINI_REQUESTS_PER_SECOND, per=1.0)

    def _smart_truncate(self, content: str, max_chars: int = 32000) -> str:
        """
        Strided sampling truncation for RST content.

        Strategy:
        - If content fits, return as-is
        - If exceeds max_chars, sample:
          - 30% from start (overview, synopsis, usage)
          - 60% from middle (main content, options, details)
          - 10% from end (examples, references, notes)
        """

        if len(content) <= max_chars:
            return content

        # Calculate segment sizes
        total_chars = max_chars - 200  # Reserve space for truncation markers
        start_chars = int(total_chars * 0.3)
        middle_chars = int(total_chars * 0.6)
        end_chars = int(total_chars * 0.1)

        # Extract segments
        content_length = len(content)

        # Start segment
        start_segment = content[:start_chars]

        # Middle segment - calculate position
        middle_start = (content_length - middle_chars) // 2
        middle_end = middle_start + middle_chars
        middle_segment = content[middle_start:middle_end]

        # End segment
        end_segment = content[-end_chars:]

        # Combine with truncation markers
        result = [
            start_segment,
            "\n\n... [START SECTION ENDS - MIDDLE SECTION BEGINS] ...\n\n",
            middle_segment,
            "\n\n... [MIDDLE SECTION ENDS - FINAL SECTION BEGINS] ...\n\n",
            end_segment,
        ]

        return "".join(result)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3)
    )
    async def analyze_rst_document(
        self, title: str, content: str, doc_type: str, path: str
    ) -> DocumentAnalysis:
        """
        Analyze RST document and extract structured information.

        RST format is similar to Markdown but with different syntax:
        - Headers use underlines (===, ---, ~~~)
        - Directives use .. syntax
        - References use :ref: or [Name]_
        - Code blocks use :: or .. code-block::
        """

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Smart content truncation to preserve key sections
        # Increased to 32K - only 1 file out of 186 exceeds this
        content_preview = self._smart_truncate(content, 32000)

        prompt = f"""
        Analyze this MRtrix3 documentation in reStructuredText (RST) format.

        Title: {title}
        Document Type Hint: {doc_type}
        Path: {path}

        RST Content:
        {content_preview}

        Please extract:
        1. Document type: MUST be one of exactly these four: [command, guide, tutorial, reference]
           - Use "command" for tool documentation
           - Use "tutorial" for step-by-step guides
           - Use "reference" for API/technical references
           - Use "guide" for concepts, installation, or general documentation
        2. Command name: If this is a command documentation, extract the exact command name
        3. Key concepts: List 3-5 main technical concepts covered
        4. Synopsis: 2-4 sentence summary of what this document/command does
        5. Command usage: If it's a command, extract the basic usage pattern

        Note: This is RST format where:
        - Headers are text with underline characters (=, -, ~)
        - Code blocks start with :: or .. code-block::
        - References look like [Name]_ or :ref:`target`
        - Directives start with ..

        Return as JSON:
        {{
            "doc_type": "...",
            "command_name": "..." or null,
            "concepts": ["concept1", "concept2", ...],
            "synopsis": "...",
            "command_usage": "..." or null
        }}
        """

        response = await asyncio.to_thread(
            self.model.generate_content, prompt, generation_config={"temperature": 0.2}
        )

        # Parse JSON from response
        try:
            # Extract JSON from response
            text = response.text.strip()
            # Handle code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)

            # Validate and sanitize the response
            validated_data = self._validate_analysis(data, doc_type, title, path)
            return DocumentAnalysis(**validated_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse Gemini response for {path}: {e}")
            # Return default analysis
            return DocumentAnalysis(
                doc_type=doc_type,
                command_name=None,
                concepts=[],
                synopsis=title[:200] if len(title) > 200 else title,
                command_usage=None,
            )

    def _validate_analysis(
        self, data: dict, fallback_doc_type: str, title: str, path: str
    ) -> dict:
        """
        Validate and sanitize Gemini's response to ensure database constraints.
        """

        # Ensure doc_type is valid
        valid_doc_types = ["command", "guide", "tutorial", "reference"]
        if "doc_type" not in data or data["doc_type"] not in valid_doc_types:
            logger.warning(
                f"Invalid doc_type '{data.get('doc_type')}' for {path}, using fallback"
            )
            data["doc_type"] = fallback_doc_type

        # For non-commands, ensure command fields are null
        if data["doc_type"] != "command":
            data["command_name"] = None
            data["command_usage"] = None

        # Validate command_name for commands
        if data["doc_type"] == "command":
            if not data.get("command_name"):
                # Try to extract from path
                if "commands/" in path:
                    data["command_name"] = Path(path).stem
                else:
                    logger.warning(f"Command doc without command_name: {path}")

        # Ensure concepts is a list
        if not isinstance(data.get("concepts"), list):
            data["concepts"] = []

        # Limit concepts to avoid database overflow
        data["concepts"] = data.get("concepts", [])[:10]

        # Ensure synopsis exists and is reasonable length
        if not data.get("synopsis"):
            data["synopsis"] = f"Documentation for {title}"
        elif len(data["synopsis"]) > 500:
            data["synopsis"] = data["synopsis"][:497] + "..."

        # Validate command_usage length
        if data.get("command_usage") and len(data["command_usage"]) > 500:
            data["command_usage"] = data["command_usage"][:497] + "..."

        return data


class ErrorExtractor:
    """Extract error patterns from MRtrix3 source code"""

    def __init__(self, source_path: Path):
        """Initialize with path to MRtrix3 source (from sparse checkout)"""
        self.source_path = source_path

    def extract_from_command(self, command_name: str) -> Optional[List[str]]:
        """Extract error messages for a specific command"""
        errors = []

        # Look for C++ command file
        cpp_file = self.source_path / "cpp" / "cmd" / f"{command_name}.cpp"
        if cpp_file.exists():
            errors.extend(self.extract_from_file(cpp_file))

        # Look for Python command file
        py_file = (
            self.source_path / "python" / "mrtrix3" / "commands" / f"{command_name}.py"
        )
        if py_file.exists():
            errors.extend(self.extract_from_file(py_file))

        return errors if errors else None

    def extract_from_file(self, file_path: Path) -> List[str]:
        """Extract error patterns from a source file"""
        errors = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # C++ error patterns
            cpp_patterns = [
                r'throw\s+Exception\s*\(\s*"([^"]+)"',
                r'WARN\s*\(\s*"([^"]+)"',
                r'ERROR\s*\(\s*"([^"]+)"',
            ]

            # Python error patterns
            py_patterns = [
                r'raise\s+\w+Error\s*\(\s*["\']([^"\']+)["\']',
                r'app\.error\s*\(\s*["\']([^"\']+)["\']',
                r'app\.warn\s*\(\s*["\']([^"\']+)["\']',
            ]

            patterns = cpp_patterns if file_path.suffix == ".cpp" else py_patterns

            for pattern in patterns:
                matches = re.findall(pattern, content)
                errors.extend(matches)

        except Exception as e:
            logger.warning(f"Error extracting from {file_path}: {e}")

        return list(set(errors))  # Remove duplicates


class SupabaseManager:
    """Manage Supabase database operations"""

    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def get_existing_urls(self) -> set:
        """Get URLs already in database"""
        result = self.client.table("documents").select("source_url").execute()
        return {doc["source_url"] for doc in result.data}

    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into database"""
        try:
            # Add UUIDs
            for doc in documents:
                doc["id"] = str(uuid4())

            self.client.table("documents").insert(documents).execute()
            return True
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            return False


class DocumentProcessor:
    """Main processor orchestrating the pipeline"""

    def __init__(self):
        self.gatherer = RSTDocumentGatherer()
        self.analyzer = GeminiRSTAnalyzer()
        self.error_extractor = None  # Will be initialized with repo path
        self.db_manager = SupabaseManager()

    async def process_document(
        self, doc: Dict[str, Any], version: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single RST document"""

        # Skip if very short
        if len(doc["content"]) < 100:
            return None

        # Analyze with Gemini
        analysis = await self.analyzer.analyze_rst_document(
            doc["title"], doc["content"], doc["doc_type"], doc["path"]
        )

        # Extract errors ONLY if it's a command type (database constraint)
        error_types = None
        if analysis.doc_type == "command" and analysis.command_name:
            error_types = self.error_extractor.extract_from_command(
                analysis.command_name
            )

        # Build document record
        now = datetime.now(timezone.utc).isoformat()
        return {
            "title": doc["title"],
            "content": doc["content"],  # Store raw RST
            "source_url": doc["url"],
            "doc_type": analysis.doc_type,
            "command_name": analysis.command_name
            if analysis.doc_type == "command"
            else None,  # Only for commands
            "concepts": analysis.concepts,
            "synopsis": analysis.synopsis,
            "command_usage": analysis.command_usage
            if analysis.doc_type == "command"
            else None,  # Only for commands
            "error_types": error_types,
            "version": version,
            "last_updated": now,
            "created_at": now,
            "updated_at": now,
        }

    async def run(self, batch_size: Optional[int] = None, skip_existing: bool = True):
        """Main execution method"""

        console.print("[bold cyan]MRtrix3 Documentation Processor")
        console.print("[dim]Using sparse checkout for complete coverage\n")

        # Use temp directory for the entire processing lifecycle
        with tempfile.TemporaryDirectory() as tmpdir:
            # Gather documents
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Gathering RST documents...", total=None)
                documents, version, repo_path = (
                    self.gatherer.gather_documents_and_source(tmpdir)
                )
                progress.remove_task(task)

            # Initialize error extractor with temporary repo path
            self.error_extractor = ErrorExtractor(repo_path)

            console.print(
                f"[green]✓ Gathered {len(documents)} documents from version {version}"
            )

            # Filter existing if needed
            if skip_existing:
                existing_urls = self.db_manager.get_existing_urls()
                documents = [d for d in documents if d["url"] not in existing_urls]
                console.print(f"[yellow]Filtered to {len(documents)} new documents")

            # Apply batch limit if specified
            if batch_size and batch_size < len(documents):
                documents = documents[:batch_size]
                console.print(f"[yellow]Limited to {batch_size} documents")

            if not documents:
                console.print("[yellow]No new documents to process")
                return

            # Process documents concurrently in batches
            processed_docs = []
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Processing documents...", total=len(documents)
                )

                # Process in concurrent batches
                for i in range(0, len(documents), GEMINI_CONCURRENT_LIMIT):
                    batch = documents[i : i + GEMINI_CONCURRENT_LIMIT]

                    # Create tasks for concurrent processing
                    tasks = [self.process_document(doc, version) for doc in batch]

                    # Wait for all tasks in batch to complete
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle results
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.warning(f"Error processing document: {result}")
                        elif result:
                            processed_docs.append(result)
                        progress.advance(task)

            # Insert into database
            if processed_docs:
                console.print(
                    f"\n[cyan]Inserting {len(processed_docs)} documents into database..."
                )
                if self.db_manager.insert_documents(processed_docs):
                    console.print("[green]✓ Successfully inserted documents")
                else:
                    console.print("[red]✗ Failed to insert documents")

            # Show summary
            self.show_summary(processed_docs)

    def show_summary(self, documents: List[Dict[str, Any]]):
        """Display processing summary"""
        if not documents:
            return

        from collections import Counter

        doc_types = Counter(doc["doc_type"] for doc in documents)

        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Total Documents", str(len(documents)))
        for dtype, count in doc_types.most_common():
            table.add_row(f"  - {dtype}", str(count))

        console.print("\n", table)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Populate MRtrix3 RAG database using sparse checkout"
    )
    parser.add_argument("--batch-size", type=int, help="Process only N documents")
    parser.add_argument(
        "--include-existing", action="store_true", help="Reprocess existing URLs"
    )
    args = parser.parse_args()

    processor = DocumentProcessor()
    await processor.run(
        batch_size=args.batch_size, skip_existing=not args.include_existing
    )


if __name__ == "__main__":
    asyncio.run(main())
