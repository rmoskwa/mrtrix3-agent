# Schema Migration Guide for Developers

This guide explains how to add new schema versions to the ChromaDB local storage.

## 📋 Quick Steps to Add a New Migration

When you need to change the ChromaDB schema (add fields, modify structure, etc.):

### 1️⃣ Open `src/agent/schema_migrations.py`

### 2️⃣ Update the Version
```python
# Line ~30
CURRENT_VERSION = "1.1.0"  # Update from "1.0.0"
```

### 3️⃣ Add New Fields (if needed)
```python
# Line ~35
METADATA_FIELDS = {
    ...existing fields...,
    'complexity': 'string',  # Add your new field
}
```

### 4️⃣ Write Your Migration Function
```python
# After line ~65
def migrate_v1_0_to_v1_1(collection, manager):
    """Migrate from v1.0.0 to v1.1.0 - Add complexity field."""

    logger.info("Starting migration v1.0.0 -> v1.1.0")

    # Get all documents
    result = collection.get(include=['metadatas'])

    # Update each document
    for i, doc_id in enumerate(result['ids']):
        metadata = result['metadatas'][i] or {}

        # Add new field with default value
        if 'complexity' not in metadata:
            metadata['complexity'] = 'medium'

            # Update the document
            collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )

    logger.info(f"Updated {len(result['ids'])} documents")
```

### 5️⃣ Register the Migration
```python
# Line ~100
MIGRATIONS: List[Migration] = [
    ...existing migrations...,
    Migration(
        from_version="1.0.0",
        to_version="1.1.0",
        migrate_func=migrate_v1_0_to_v1_1,
        description="Add complexity field for better search"
    ),
]
```

## ✅ That's It!

The migration will automatically run when users start the application with an older database version.

## 🧪 Testing Your Migration

### Test with Existing Data
```bash
# 1. Create a backup of your current ChromaDB
cp -r ~/.mrtrix3-agent/chromadb ~/.mrtrix3-agent/chromadb_backup

# 2. Run the application - migration should trigger
python -m src.agent.cli

# 3. Check the logs for migration messages
# You should see: "Running migration: 1.0.0 -> 1.1.0"
```

### Test Migration Path
```python
# Test script
from src.agent.local_storage_manager import LocalDatabaseManager

manager = LocalDatabaseManager("~/.mrtrix3-agent/chromadb")
collection = manager.initialize_collection()

# Check version
print(f"Schema version: {collection.metadata.get('schema_version')}")

# Verify your changes
result = collection.get(limit=1, include=['metadatas'])
print(f"Sample metadata: {result['metadatas'][0]}")
```

## 📊 Migration Examples

### Example 1: Adding a Field
```python
def migrate_add_field(collection, manager):
    """Add a new metadata field to all documents."""
    result = collection.get(include=['metadatas'])

    for i, doc_id in enumerate(result['ids']):
        metadata = result['metadatas'][i] or {}
        metadata['new_field'] = 'default_value'

        collection.update(
            ids=[doc_id],
            metadatas=[metadata]
        )
```

### Example 2: Renaming a Field
```python
def migrate_rename_field(collection, manager):
    """Rename 'old_field' to 'new_field'."""
    result = collection.get(include=['metadatas'])

    for i, doc_id in enumerate(result['ids']):
        metadata = result['metadatas'][i] or {}

        if 'old_field' in metadata:
            metadata['new_field'] = metadata.pop('old_field')

            collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
```

### Example 3: Transforming Data
```python
def migrate_transform_data(collection, manager):
    """Transform existing data format."""
    result = collection.get(include=['metadatas'])

    for i, doc_id in enumerate(result['ids']):
        metadata = result['metadatas'][i] or {}

        # Example: Convert string list to JSON array
        if 'keywords' in metadata and isinstance(metadata['keywords'], str):
            metadata['keywords'] = json.dumps(metadata['keywords'].split(','))

            collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
```

## ⚠️ Important Notes

1. **Migrations are permanent** - Once a database is migrated, it cannot be downgraded
2. **Test thoroughly** - Always test with a backup first
3. **Chain migrations** - Users can skip versions (1.0.0 → 1.5.0), the system will run all intermediate migrations
4. **Document changes** - Update this guide with any complex migrations for future reference

## 🔍 How It Works

The migration system:
1. Checks the current database version on startup
2. Compares with the code's `CURRENT_VERSION`
3. If different, finds the migration path
4. Runs each migration in sequence
5. Updates the database metadata with the new version

## 📝 Version Numbering

Follow semantic versioning:
- **Major** (1.0.0 → 2.0.0): Breaking changes, incompatible schema
- **Minor** (1.0.0 → 1.1.0): New fields, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, no schema changes

## 🚨 Troubleshooting

### Migration Fails
```python
# Check the error in logs
# Restore from backup
rm -rf ~/.mrtrix3-agent/chromadb
mv ~/.mrtrix3-agent/chromadb_backup ~/.mrtrix3-agent/chromadb
```

### Missing Migration Path
```
# Error: No migration path from X to Y
# Solution: Add intermediate migrations in MIGRATIONS list
```

### Data Corruption
```python
# Always backup before migration
# Use atomic operations when possible
# Test with small datasets first
```

## 📚 Resources

- ChromaDB Documentation: https://docs.trychroma.com/
- Migration File: `src/agent/schema_migrations.py`
- Storage Manager: `src/agent/local_storage_manager.py`
