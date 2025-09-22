"""Schema migration definitions for ChromaDB.

To add a new migration:
1. Increment CURRENT_VERSION
2. Add your migration function to MIGRATIONS list
3. That's it!

Example:
    CURRENT_VERSION = "1.1.0"

    def migrate_v1_0_to_v1_1(collection, manager):
        # Your migration logic
        pass

    MIGRATIONS = [
        ...existing migrations...,
        Migration("1.0.0", "1.1.0", migrate_v1_0_to_v1_1)
    ]
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DEVELOPER: UPDATE THIS VERSION WHEN ADDING NEW MIGRATIONS
# ============================================================================
CURRENT_VERSION = "1.0.0"

# ============================================================================
# SCHEMA FIELD DEFINITIONS
# ============================================================================
METADATA_FIELDS = {
    "doc_type": "string",
    "title": "string",
    "keywords": "string",
    "version": "string",
    "command_name": "string",
    "command_usage": "string",
    "error_types": "string",
    "synopsis": "string",
    "source_url": "string",
    "last_updated": "string",
    # Add new fields here when updating schema
}


@dataclass
class Migration:
    """Represents a schema migration."""

    from_version: str
    to_version: str
    migrate_func: Callable
    description: str = ""


# ============================================================================
# MIGRATION FUNCTIONS - Add new migrations here
# ============================================================================


def migrate_v0_to_v1(collection, manager):
    """Initial schema - no migration needed."""
    logger.info("Initial schema v1.0.0 - no migration needed")
    # Parameters kept for consistency with migration interface
    _ = (collection, manager)  # Unused but required by interface


# Example for next migration (uncomment and modify when needed):
# """
# def migrate_v1_0_to_v1_1(collection, manager):
#     '''Migrate from v1.0.0 to v1.1.0 - Add complexity field.'''
#
#     logger.info("Starting migration v1.0.0 -> v1.1.0")

#     # Get all documents
#     result = collection.get(include=['metadatas'])
#
#     # Update each document's metadata
#     for i, doc_id in enumerate(result['ids']):
#         metadata = result['metadatas'][i] or {}
#
#         # Add new field with default value
#         if 'complexity' not in metadata:
#             metadata['complexity'] = 'medium'
#
#             # Update the document
#             collection.update(
#                 ids=[doc_id],
#                 metadatas=[metadata]
#             )
#
#     logger.info(f"Updated {len(result['ids'])} documents with complexity field")
# """


# ============================================================================
# MIGRATION REGISTRY - Add all migrations here in order
# ============================================================================
MIGRATIONS: List[Migration] = [
    Migration(
        from_version="0.0.0",
        to_version="1.0.0",
        migrate_func=migrate_v0_to_v1,
        description="Initial schema setup",
    ),
    # Add new migrations here:
    # Migration(
    #     from_version="1.0.0",
    #     to_version="1.1.0",
    #     migrate_func=migrate_v1_0_to_v1_1,
    #     description="Add complexity field"
    # ),
]


class SchemaMigrator:
    """Handles schema migrations for ChromaDB collections."""

    def __init__(self):
        self.migrations = {(m.from_version, m.to_version): m for m in MIGRATIONS}

    def get_current_version(self) -> str:
        """Get the current schema version."""
        return CURRENT_VERSION

    def get_metadata_fields(self) -> Dict[str, str]:
        """Get the current metadata field definitions."""
        return METADATA_FIELDS

    def needs_migration(self, from_version: str) -> bool:
        """Check if migration is needed."""
        return from_version != CURRENT_VERSION

    def get_migration_path(self, from_version: str, to_version: str) -> List[Migration]:
        """Get the migration path from one version to another."""
        path = []
        current = from_version

        while current != to_version:
            found = False
            for migration in MIGRATIONS:
                if migration.from_version == current:
                    path.append(migration)
                    current = migration.to_version
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"No migration path from {from_version} to {to_version}"
                )

        return path

    def migrate(
        self, collection: Any, manager: Any, from_version: str, to_version: str
    ):
        """Execute migrations from one version to another."""
        if from_version == to_version:
            logger.info(f"Already at version {to_version}, no migration needed")
            return

        migrations = self.get_migration_path(from_version, to_version)

        for migration in migrations:
            logger.info(
                f"Running migration: {migration.from_version} -> {migration.to_version}"
            )
            if migration.description:
                logger.info(f"  Description: {migration.description}")

            try:
                migration.migrate_func(collection, manager)
                logger.info("  ✓ Migration completed successfully")
            except Exception as e:
                logger.error(f"  ✗ Migration failed: {e}")
                raise

        # Update collection metadata with new version
        collection.modify(metadata={"schema_version": to_version})
        logger.info(f"Schema updated to version {to_version}")


# ============================================================================
# MIGRATION GUIDE FOR DEVELOPERS
# ============================================================================
# HOW TO ADD A NEW SCHEMA MIGRATION:
#
# 1. Update CURRENT_VERSION at the top of this file
#    Example: CURRENT_VERSION = "1.1.0"
#
# 2. Add new fields to METADATA_FIELDS if needed
#    Example: 'complexity': 'string'
#
# 3. Write your migration function
#    Example:
#    def migrate_v1_0_to_v1_1(collection, manager):
#        # Your migration logic here
#        pass
#
# 4. Add the migration to MIGRATIONS list
#    Example:
#    Migration(
#        from_version="1.0.0",
#        to_version="1.1.0",
#        migrate_func=migrate_v1_0_to_v1_1,
#        description="Add complexity field"
#    )
#
# That's it! The migration will run automatically when users update.
#
# TESTING YOUR MIGRATION:
# - Test with an existing database at the old version
# - Verify data is preserved and transformed correctly
# - Check that re-running doesn't cause issues
