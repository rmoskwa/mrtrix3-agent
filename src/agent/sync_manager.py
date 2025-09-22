"""Database sync manager for MRtrix3 agent - handles Supabase to ChromaDB synchronization."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import shutil

import chromadb
from chromadb.api import ClientAPI
from supabase import Client
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.prompt import Prompt

from .models import AgentConfiguration


class SyncMetadata:
    """Manages sync metadata for tracking database state."""

    def __init__(self, storage_path: str):
        """Initialize sync metadata manager.

        Args:
            storage_path: Path to ChromaDB storage directory.
        """
        self.storage_path = Path(storage_path)
        self.metadata_file = self.storage_path / "sync_metadata.json"
        self.preferences_file = self.storage_path / "user_preferences.json"

    def get_last_sync_hash(self) -> Optional[str]:
        """Get the hash from the last successful sync."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    return data.get("last_sync_hash")
            except Exception:
                return None
        return None

    def update_sync_metadata(self, sync_hash: str, document_count: int):
        """Update metadata after successful sync.

        Args:
            sync_hash: Hash of the synced database content.
            document_count: Number of documents synced.
        """
        metadata = {
            "last_sync_hash": sync_hash,
            "last_sync_time": datetime.now().isoformat(),
            "document_count": document_count,
        }

        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_user_preferences(self) -> Dict:
        """Get user preferences for sync behavior."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def update_user_preferences(self, preferences: Dict):
        """Update user preferences.

        Args:
            preferences: Dictionary of user preferences.
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self.preferences_file, "w") as f:
            json.dump(preferences, f, indent=2)

    def increment_app_uses(self):
        """Increment the application use counter."""
        prefs = self.get_user_preferences()
        prefs["app_uses_since_reminder"] = prefs.get("app_uses_since_reminder", 0) + 1
        self.update_user_preferences(prefs)


class DatabaseSyncManager:
    """Manages synchronization between Supabase and ChromaDB."""

    def __init__(
        self,
        supabase_client: Client,
        chromadb_client: ClientAPI,
        chromadb_path: str,
        config: Optional[AgentConfiguration] = None,
    ):
        """Initialize the sync manager.

        Args:
            supabase_client: Supabase client for fetching data.
            chromadb_client: ChromaDB client for local storage.
            chromadb_path: Path to ChromaDB storage directory.
            config: Optional agent configuration.
        """
        self.supabase_client = supabase_client
        self.chromadb_client = chromadb_client
        self.chromadb_path = chromadb_path
        self.config = config or AgentConfiguration()
        self.metadata = SyncMetadata(chromadb_path)
        self.console = Console()

    def calculate_database_hash(self, documents: List[Dict]) -> str:
        """Calculate hash of database content for change detection.

        Args:
            documents: List of document dictionaries.

        Returns:
            SHA-256 hash of the sorted document content.
        """
        # Sort documents by ID for consistent hashing
        sorted_docs = sorted(documents, key=lambda x: x.get("id", ""))

        # Create a stable string representation
        content_str = json.dumps(sorted_docs, sort_keys=True, default=str)

        # Calculate SHA-256 hash
        return hashlib.sha256(content_str.encode()).hexdigest()

    def fetch_documents_from_supabase(
        self, show_progress: bool = True
    ) -> Tuple[List[Dict], str]:
        """Fetch all documents from Supabase.

        Args:
            show_progress: Whether to show progress indicator.

        Returns:
            Tuple of (documents list, content hash).
        """
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Fetching documents from Supabase...", total=None
                )

                # Fetch all documents from the documents table
                response = self.supabase_client.table("documents").select("*").execute()
                documents = response.data

                progress.update(
                    task,
                    completed=True,
                    description=f"Fetched {len(documents)} documents",
                )
        else:
            response = self.supabase_client.table("documents").select("*").execute()
            documents = response.data

        # Calculate hash of the fetched content
        content_hash = self.calculate_database_hash(documents)

        return documents, content_hash

    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if updates are available from Supabase.

        Returns:
            Tuple of (has_updates, current_hash, new_hash).
        """
        try:
            # Get current local hash
            current_hash = self.metadata.get_last_sync_hash()

            # Fetch documents and calculate new hash
            self.console.print("🔍 Checking for updates...")
            documents, new_hash = self.fetch_documents_from_supabase(
                show_progress=False
            )

            # Compare hashes
            has_updates = current_hash != new_hash

            if has_updates:
                self.console.print(f"📦 Updates available: {len(documents)} documents")
            else:
                self.console.print("✅ Local database is up to date")

            return has_updates, current_hash, new_hash

        except Exception as e:
            self.console.print(f"⚠️  Update check failed: {str(e)}")
            self.console.print("Continuing with local data...")
            return False, None, None

    def should_prompt_for_update(self) -> bool:
        """Check if user should be prompted for update based on preferences.

        Returns:
            True if user should be prompted, False otherwise.
        """
        prefs = self.metadata.get_user_preferences()

        # Check if user selected "don't remind"
        if prefs.get("update_preference") == "never":
            return False

        # Check if user selected "remind me later" and threshold reached
        if prefs.get("update_preference") == "later":
            uses = prefs.get("app_uses_since_reminder", 0)
            if uses >= 5:
                # Reset counter and prompt
                prefs["app_uses_since_reminder"] = 0
                self.metadata.update_user_preferences(prefs)
                return True
            else:
                # Increment counter but don't prompt
                return False

        # Default: always prompt
        return True

    def prompt_for_update(self) -> str:
        """Prompt user for update preference.

        Returns:
            User's choice: 'now', 'later', or 'never'.
        """
        self.console.print("\n📊 [bold yellow]Database Update Available[/bold yellow]")
        self.console.print(
            "The MRtrix3 documentation has been updated since your last sync.\n"
        )

        choices = {
            "1": ("now", "Update now"),
            "2": ("later", "Remind me later (after 5 uses)"),
            "3": ("never", "Don't remind me about this update"),
        }

        for key, (_, label) in choices.items():
            self.console.print(f"  {key}. {label}")

        choice = Prompt.ask("\nYour choice", choices=["1", "2", "3"], default="1")

        action, _ = choices[choice]

        # Save preference
        if action in ["later", "never"]:
            prefs = self.metadata.get_user_preferences()
            prefs["update_preference"] = action
            if action == "later":
                prefs["app_uses_since_reminder"] = 0
            self.metadata.update_user_preferences(prefs)

        return action

    def populate_chromadb(
        self,
        documents: List[Dict],
        collection_name: str = "mrtrix3_documents",
        show_progress: bool = True,
    ):
        """Populate ChromaDB collection with documents.

        Args:
            documents: List of document dictionaries from Supabase.
            collection_name: Name of the ChromaDB collection.
            show_progress: Whether to show progress bar.
        """
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        contents = []

        for doc in documents:
            # Parse embedding if it's a string (field is 'content_embedding' in Supabase)
            embedding = doc.get("content_embedding")
            if not embedding:
                # Skip documents without embeddings
                continue

            if isinstance(embedding, str):
                embedding = json.loads(embedding)

            # Only add documents that have embeddings
            ids.append(str(doc["id"]))
            embeddings.append(embedding)

            # Prepare metadata
            metadata = {
                "doc_type": doc.get("doc_type", ""),
                "title": doc.get("title", ""),
                "keywords": doc.get("keywords", ""),
                "version": doc.get("version", "latest"),
                "source_url": doc.get("source_url", ""),
                "synopsis": doc.get("synopsis", ""),
            }

            # Add optional fields if present
            if doc.get("command_name"):
                metadata["command_name"] = doc["command_name"]
            if doc.get("command_usage"):
                metadata["command_usage"] = doc["command_usage"]
            if doc.get("error_types"):
                metadata["error_types"] = (
                    str(doc["error_types"]) if doc["error_types"] else ""
                )

            metadatas.append(metadata)
            contents.append(doc.get("content", ""))

        # Add to collection with progress bar
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Populating ChromaDB...", total=len(documents))

                # Get or create collection
                collection = self.chromadb_client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "description": "MRtrix3 documentation for semantic search",
                        "embedding_dimensions": "768",
                        "source": "Supabase sync",
                        "last_sync": datetime.now().isoformat(),
                    },
                )

                # Add documents in batches to show progress
                batch_size = 50
                for i in range(0, len(ids), batch_size):
                    end_idx = min(i + batch_size, len(ids))

                    collection.add(
                        ids=ids[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        documents=contents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                    )

                    progress.update(task, completed=end_idx)
        else:
            collection = self.chromadb_client.get_or_create_collection(
                name=collection_name
            )
            collection.add(
                ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas
            )

    def perform_atomic_sync(self, documents: List[Dict], content_hash: str) -> bool:
        """Perform atomic database replacement.

        Args:
            documents: Documents to sync.
            content_hash: Hash of the content being synced.

        Returns:
            True if successful, False otherwise.
        """
        temp_path = None
        try:
            # Create temporary ChromaDB instance
            temp_path = tempfile.mkdtemp(prefix="chromadb_sync_")
            temp_client = chromadb.PersistentClient(
                path=temp_path, settings=chromadb.Settings(anonymized_telemetry=False)
            )

            # Populate temporary database
            self.console.print("📥 Downloading and processing documents...")
            temp_client.get_or_create_collection("mrtrix3_documents")

            # Temporarily swap client for population
            original_client = self.chromadb_client
            self.chromadb_client = temp_client
            self.populate_chromadb(documents, show_progress=True)
            self.chromadb_client = original_client

            # Atomic swap: backup old, move new
            self.console.print("🔄 Performing atomic database swap...")

            chromadb_dir = Path(self.chromadb_path)
            backup_path = chromadb_dir.parent / f"{chromadb_dir.name}_backup"

            # Remove old backup if exists
            if backup_path.exists():
                shutil.rmtree(backup_path)

            # Move current to backup (if exists)
            if chromadb_dir.exists():
                shutil.move(str(chromadb_dir), str(backup_path))

            # Move temp to production
            shutil.move(temp_path, str(chromadb_dir))

            # Update metadata
            self.metadata.update_sync_metadata(content_hash, len(documents))

            # Remove backup after successful swap
            if backup_path.exists():
                shutil.rmtree(backup_path)

            self.console.print("✅ Database successfully updated!")
            return True

        except Exception as e:
            self.console.print(f"❌ Sync failed: {str(e)}")

            # Attempt to restore backup
            chromadb_dir = Path(self.chromadb_path)
            backup_path = chromadb_dir.parent / f"{chromadb_dir.name}_backup"
            if backup_path.exists() and not chromadb_dir.exists():
                shutil.move(str(backup_path), str(chromadb_dir))
                self.console.print("↩️  Restored previous database")

            return False

        finally:
            # Clean up temp directory if it still exists
            if temp_path and Path(temp_path).exists():
                shutil.rmtree(temp_path)

    def sync_on_startup(self) -> bool:
        """Main sync method to be called on application startup.

        Returns:
            True if sync was performed or skipped by user, False on error.
        """
        try:
            # Increment app use counter
            self.metadata.increment_app_uses()

            # Check if this is first run (no metadata file)
            is_first_run = not self.metadata.get_last_sync_hash()

            if is_first_run:
                self.console.print(
                    "\n🎉 [bold green]Welcome to MRtrix3 Agent![/bold green]"
                )
                self.console.print("This appears to be your first run.")
                self.console.print(
                    "I'll download the MRtrix3 documentation database now...\n"
                )

                # Fetch and sync
                documents, content_hash = self.fetch_documents_from_supabase()
                return self.perform_atomic_sync(documents, content_hash)

            # Check for updates
            has_updates, current_hash, new_hash = self.check_for_updates()

            if not has_updates:
                return True

            # Check if we should prompt
            if not self.should_prompt_for_update():
                self.console.print("📝 Skipping update based on user preference")
                return True

            # Prompt user
            choice = self.prompt_for_update()

            if choice == "now":
                # Fetch full documents and sync
                documents, content_hash = self.fetch_documents_from_supabase()
                return self.perform_atomic_sync(documents, content_hash)
            else:
                self.console.print("📝 Update deferred")
                return True

        except Exception as e:
            self.console.print(f"❌ Sync error: {str(e)}")
            self.console.print("Continuing with local database...")
            return False
