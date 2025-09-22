"""Local storage management for ChromaDB - handles schema, health, and maintenance."""

import os
import json
import psutil
import fcntl
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import chromadb
from chromadb.api import ClientAPI
from chromadb import Collection
from rich.console import Console
from rich.table import Table
import logging

from .schema_migrations import SchemaMigrator

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class StorageStats:
    """Storage statistics for ChromaDB."""

    total_size_mb: float
    document_count: int
    collection_count: int
    temp_files_count: int
    temp_files_size_mb: float
    last_sync_time: Optional[str]
    database_version: str
    schema_version: str


@dataclass
class SchemaDefinition:
    """Schema definition for ChromaDB collection."""

    collection_name: str = "mrtrix3_documents"

    def __init__(self):
        self.migrator = SchemaMigrator()
        self.version = self.migrator.get_current_version()
        self.metadata_fields = self.migrator.get_metadata_fields()


class LockManager:
    """Manages file-based locking to prevent concurrent access."""

    def __init__(self, storage_path: str):
        """Initialize lock manager.

        Args:
            storage_path: Path to ChromaDB storage directory.
        """
        self.storage_path = Path(storage_path)
        self.lock_file = self.storage_path / ".lock"
        self.pid_file = self.storage_path / ".pid"
        self.lock_fd = None

    def acquire_lock(self, timeout: int = 5) -> bool:
        """Acquire exclusive lock on ChromaDB directory.

        Args:
            timeout: Maximum time to wait for lock in seconds.

        Returns:
            True if lock acquired, False otherwise.
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)

        try:
            # Open lock file
            self.lock_fd = open(self.lock_file, "w")

            # Try to acquire exclusive lock
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID
            self.lock_fd.write(str(os.getpid()))
            self.lock_fd.flush()

            # Also write PID to separate file for debugging
            with open(self.pid_file, "w") as f:
                f.write(f"{os.getpid()}\n{datetime.now().isoformat()}")

            return True

        except (IOError, OSError):
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None

            # Check if the process holding the lock is still alive
            if self.pid_file.exists():
                try:
                    with open(self.pid_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            old_pid = int(lines[0].strip())
                            if not self._is_process_running(old_pid):
                                # Process is dead, clean up lock
                                logger.warning(
                                    f"Cleaning up stale lock from PID {old_pid}"
                                )
                                self.release_lock(force=True)
                                return self.acquire_lock(timeout=0)
                except (ValueError, IOError):
                    pass

            return False

    def release_lock(self, force: bool = False):
        """Release the lock.

        Args:
            force: Force release even if not owned by current process.
        """
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                self.lock_fd.close()
            except Exception:
                pass
            self.lock_fd = None

        if force or self._owns_lock():
            try:
                self.lock_file.unlink(missing_ok=True)
                self.pid_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _owns_lock(self) -> bool:
        """Check if current process owns the lock."""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        return int(lines[0].strip()) == os.getpid()
            except (ValueError, IOError):
                pass
        return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            # Fallback method
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


class LocalDatabaseManager:
    """Manages ChromaDB local storage, schema, and health."""

    def __init__(self, storage_path: str, client: Optional[ClientAPI] = None):
        """Initialize local database manager.

        Args:
            storage_path: Path to ChromaDB storage directory.
            client: Optional existing ChromaDB client.
        """
        self.storage_path = Path(storage_path)
        self.schema = SchemaDefinition()
        self.lock_manager = LockManager(storage_path)

        if client:
            self.client = client
        else:
            self.client = self._create_client()

    def _create_client(self) -> ClientAPI:
        """Create ChromaDB client."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

    def initialize_collection(self, force_migration: bool = False) -> Collection:
        """Initialize or migrate ChromaDB collection.

        Args:
            force_migration: Force schema migration even if versions match.

        Returns:
            ChromaDB collection ready for use.
        """
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=self.schema.collection_name,
                metadata={
                    "description": "MRtrix3 documentation for semantic search",
                    "schema_version": self.schema.version,
                    "created_at": datetime.now().isoformat(),
                    "embedding_dimensions": "768",
                    "source": "Supabase sync",
                },
            )

            # Check if migration needed
            current_version = (
                collection.metadata.get("schema_version", "0.0.0")
                if collection.metadata
                else "0.0.0"
            )
            if current_version != self.schema.version or force_migration:
                logger.info(
                    f"Migrating schema from {current_version} to {self.schema.version}"
                )
                self._migrate_schema(collection, current_version, self.schema.version)

            return collection

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _migrate_schema(
        self, collection: Collection, from_version: str, to_version: str
    ):
        """Perform schema migration if needed.

        Args:
            collection: ChromaDB collection to migrate.
            from_version: Current schema version.
            to_version: Target schema version.
        """
        # Use the migrator to handle the migration
        self.schema.migrator.migrate(collection, self, from_version, to_version)

    def health_check(self) -> Tuple[bool, List[str]]:
        """Perform health check on local database.

        Returns:
            Tuple of (is_healthy, list of issues).
        """
        issues = []

        try:
            # Check storage path exists and is writable
            if not self.storage_path.exists():
                issues.append("Storage path does not exist")
            elif not os.access(self.storage_path, os.W_OK):
                issues.append("Storage path is not writable")

            # Check client connection
            try:
                self.client.heartbeat()
            except Exception as e:
                issues.append(f"ChromaDB client error: {e}")

            # Check collection exists and is accessible
            try:
                collection = self.client.get_collection(self.schema.collection_name)
                count = collection.count()
                if count == 0:
                    issues.append("Collection is empty - sync may be needed")
            except Exception as e:
                issues.append(f"Collection access error: {e}")

            # Check for lock issues
            if self.lock_manager.pid_file.exists():
                with open(self.lock_manager.pid_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        pid = int(lines[0].strip())
                        if pid != os.getpid() and self.lock_manager._is_process_running(
                            pid
                        ):
                            issues.append(f"Database may be locked by PID {pid}")

            # Check disk space
            stats = shutil.disk_usage(self.storage_path)
            free_gb = stats.free / (1024**3)
            if free_gb < 0.5:  # Less than 500MB free
                issues.append(f"Low disk space: {free_gb:.2f}GB free")

        except Exception as e:
            issues.append(f"Health check error: {e}")

        is_healthy = len(issues) == 0
        return is_healthy, issues

    def recover_database(self) -> bool:
        """Attempt to recover database from issues.

        Returns:
            True if recovery successful, False otherwise.
        """
        console.print("[yellow]Attempting database recovery...[/yellow]")

        try:
            # Try to clean up stale locks
            if self.lock_manager.pid_file.exists():
                with open(self.lock_manager.pid_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        pid = int(lines[0].strip())
                        if not self.lock_manager._is_process_running(pid):
                            self.lock_manager.release_lock(force=True)
                            console.print("✅ Cleaned up stale lock")

            # Try to recreate client
            self.client = self._create_client()

            # Try to reinitialize collection
            self.initialize_collection()

            # Verify recovery
            is_healthy, issues = self.health_check()

            if is_healthy:
                console.print("[green]✅ Database recovery successful![/green]")
                return True
            else:
                console.print(
                    f"[red]❌ Recovery incomplete. Remaining issues: {issues}[/red]"
                )
                return False

        except Exception as e:
            console.print(f"[red]❌ Recovery failed: {e}[/red]")
            return False

    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics.

        Returns:
            StorageStats object with current statistics.
        """
        # Calculate storage size
        total_size = 0
        temp_size = 0
        temp_count = 0

        for path in self.storage_path.rglob("*"):
            if path.is_file():
                size = path.stat().st_size
                total_size += size

                # Check for temp files
                if "tmp" in path.name.lower() or "temp" in path.name.lower():
                    temp_size += size
                    temp_count += 1

        # Get document count
        doc_count = 0
        try:
            collection = self.client.get_collection(self.schema.collection_name)
            doc_count = collection.count()
        except Exception:
            pass

        # Get collection count
        try:
            collections = self.client.list_collections()
            collection_count = len(collections)
        except Exception:
            collection_count = 0

        # Get last sync time
        sync_metadata_file = self.storage_path / "sync_metadata.json"
        last_sync = None
        db_version = "unknown"

        if sync_metadata_file.exists():
            try:
                with open(sync_metadata_file, "r") as f:
                    data = json.load(f)
                    last_sync = data.get("last_sync_time")
                    db_version = data.get("version", "unknown")
            except Exception:
                pass

        return StorageStats(
            total_size_mb=total_size / (1024 * 1024),
            document_count=doc_count,
            collection_count=collection_count,
            temp_files_count=temp_count,
            temp_files_size_mb=temp_size / (1024 * 1024),
            last_sync_time=last_sync,
            database_version=db_version,
            schema_version=self.schema.version,
        )

    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """Clean up temporary files older than specified hours.

        Args:
            older_than_hours: Remove temp files older than this many hours.

        Returns:
            Number of files cleaned up.
        """
        cleaned = 0
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        # Look for temp directories
        for path in self.storage_path.rglob("*"):
            if path.is_dir() and (
                "tmp" in path.name.lower() or "temp" in path.name.lower()
            ):
                try:
                    # Check modification time
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime < cutoff_time:
                        shutil.rmtree(path)
                        cleaned += 1
                        logger.info(f"Cleaned up temp directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean {path}: {e}")

        # Look for backup directories
        for path in self.storage_path.parent.glob(f"{self.storage_path.name}_backup*"):
            if path.is_dir():
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime < cutoff_time:
                        shutil.rmtree(path)
                        cleaned += 1
                        logger.info(f"Cleaned up backup directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean {path}: {e}")

        return cleaned

    def display_storage_info(self):
        """Display storage information in a formatted table."""
        stats = self.get_storage_stats()

        table = Table(title="ChromaDB Storage Information", show_header=True)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")

        table.add_row("Storage Path", str(self.storage_path))
        table.add_row("Total Size", f"{stats.total_size_mb:.2f} MB")
        table.add_row("Documents", str(stats.document_count))
        table.add_row("Collections", str(stats.collection_count))
        table.add_row("Schema Version", stats.schema_version)
        table.add_row("Database Version", stats.database_version)

        if stats.last_sync_time:
            table.add_row("Last Sync", stats.last_sync_time)
        else:
            table.add_row("Last Sync", "Never")

        if stats.temp_files_count > 0:
            table.add_row(
                "Temp Files",
                f"{stats.temp_files_count} files ({stats.temp_files_size_mb:.2f} MB)",
            )

        # Check health
        is_healthy, issues = self.health_check()
        if is_healthy:
            table.add_row("Health Status", "✅ Healthy")
        else:
            table.add_row("Health Status", f"⚠️ Issues: {len(issues)}")
            for issue in issues:
                table.add_row("  Issue", issue)

        console.print(table)

    def validate_document(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a document against the schema.

        Args:
            document: Document to validate.

        Returns:
            Tuple of (is_valid, list of validation errors).
        """
        errors = []

        # Check required fields
        if "id" not in document:
            errors.append("Missing required field: id")

        if "content" not in document:
            errors.append("Missing required field: content")

        if "content_embedding" not in document and "embedding" not in document:
            errors.append("Missing required field: embedding")

        # Validate embedding dimensions if present
        embedding = document.get("content_embedding") or document.get("embedding")
        if embedding:
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    errors.append("Invalid embedding format: not valid JSON")

            if isinstance(embedding, list) and len(embedding) != 768:
                errors.append(
                    f"Invalid embedding dimensions: expected 768, got {len(embedding)}"
                )

        # Validate metadata fields
        for field, field_type in self.schema.metadata_fields.items():
            if field in document:
                value = document[field]
                if field_type == "string" and not isinstance(value, str):
                    errors.append(
                        f"Field {field} should be string, got {type(value).__name__}"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors
