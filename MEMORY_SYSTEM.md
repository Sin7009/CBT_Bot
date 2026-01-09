# Agent Memory Storage System

## Overview

The CBT Bot now includes a markdown-based memory storage system that persists agent interactions in human-readable files. This system complements the existing Redis-based conversation history by providing long-term storage with rich metadata about each therapeutic session.

## Features

- **Markdown Format**: All memories are stored in human-readable markdown files
- **Per-User Storage**: Each user has their own dedicated memory file
- **Rich Metadata**: Stores not just conversations but also:
  - Emotional state analysis
  - Cognitive distortion identification
  - Thought level classification (Automatic Thoughts, Intermediate Beliefs, Core Beliefs)
  - Therapeutic techniques used
  - Session timestamps
- **Thread-Safe**: Uses file locking to handle concurrent writes safely
- **Backward Compatible**: Works alongside Redis for real-time conversation history

## Architecture

### Components

1. **MemoryManager**: Core class handling all memory operations
   - File creation and management
   - Entry formatting and parsing
   - Thread-safe writes with file locking
   - History retrieval and search

2. **MemoryEntry**: Data structure representing a single session
   - User message and agent response
   - Psychological analysis metadata
   - Technique and distortion information
   - Timestamps

### File Structure

Memory files are stored in the `agent_memory/` directory (configurable via `MEMORY_DIR` env var):

```
agent_memory/
├── user_123456789.md
├── user_987654321.md
└── user_555555555.md
```

Each file contains markdown-formatted sessions with the following structure:

```markdown
# Memory Log for User 123456789

Created: 2026-01-09 15:00:00

---

## Session: 2026-01-09 15:05:23

### 🧠 Analysis

- **Emotion**: anxiety
- **Intensity**: 7/10
- **Thought Level**: Автоматическая мысль (АМ)
- **Primary Distortion**: Катастрофизация

### 💬 Conversation

**User**: I'm worried about my presentation tomorrow

**Agent**: What evidence do you have that supports this worry?

**Technique Used**: Socratic Questioning

---
```

## Configuration

Add these settings to your `.env` file:

```bash
# Memory storage settings
MEMORY_DIR=agent_memory
USE_MEMORY_STORAGE=true
```

### Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_DIR` | Directory to store memory files | `agent_memory` |
| `USE_MEMORY_STORAGE` | Enable/disable memory storage | `true` |

## Integration

### In bot.py

The memory manager is automatically initialized when `USE_MEMORY_STORAGE=true`:

```python
from src.memory_manager import MemoryManager

memory_manager = MemoryManager(settings.MEMORY_DIR) if settings.USE_MEMORY_STORAGE else None
```

### In agent.py

The agent automatically saves interactions to memory when a memory manager is provided:

```python
agent = CBTAgent(
    api_key=settings.OPENAI_API_KEY,
    model_therapist=settings.MODEL_THERAPIST,
    model_supervisor=settings.MODEL_SUPERVISOR,
    memory_manager=memory_manager
)
```

## API Reference

### MemoryManager

#### `__init__(memory_dir: str = "agent_memory")`
Initialize the memory manager.

#### `save_memory(entry: MemoryEntry) -> None`
Save a memory entry to the user's file. Thread-safe with file locking.

#### `load_history(user_id: str, limit: int = 10) -> List[Dict[str, str]]`
Load conversation history for a user. Returns messages in OpenAI format.

#### `get_user_stats(user_id: str) -> Dict[str, Any]`
Get statistics about a user's memory (session count, file path, etc.).

#### `clear_user_memory(user_id: str) -> bool`
Delete all memory for a user.

#### `list_users() -> List[str]`
Get list of all users with stored memories.

### MemoryEntry

Dataclass representing a session entry:

```python
@dataclass
class MemoryEntry:
    timestamp: str
    user_id: str
    user_message: str
    agent_response: str
    analysis: Optional[Dict[str, Any]] = None
    technique_used: Optional[str] = None
    thought_level: Optional[str] = None
    primary_distortion: Optional[str] = None
    emotion: Optional[str] = None
    intensity: Optional[int] = None
```

## Usage Examples

### Viewing User Memory

Memory files can be read directly as they're in markdown format:

```bash
cat agent_memory/user_123456789.md
```

### Analyzing Sessions

Since files are in markdown, you can use standard text tools:

```bash
# Count total sessions for a user
grep -c "## Session:" agent_memory/user_123456789.md

# Find all catastrophizing distortions
grep "Катастрофизация" agent_memory/*.md

# Search for specific emotions
grep "anxiety" agent_memory/*.md
```

### Programmatic Access

```python
from src.memory_manager import MemoryManager

manager = MemoryManager()

# Get user stats
stats = await manager.get_user_stats("123456789")
print(f"Total sessions: {stats['total_sessions']}")

# Load recent history
history = await manager.load_history("123456789", limit=5)

# List all users
users = manager.list_users()
```

## Benefits

1. **Persistence**: Memories survive Redis restarts and bot restarts
2. **Transparency**: Human-readable format allows easy inspection and analysis
3. **Portability**: Markdown files can be easily backed up, shared, or analyzed with external tools
4. **Rich Context**: Stores complete therapeutic context, not just conversation text
5. **Research Value**: Enables analysis of patterns, techniques, and outcomes over time

## Data Privacy

⚠️ **Important**: Memory files contain sensitive therapeutic conversations. Ensure:

1. The `agent_memory/` directory is excluded from version control (already in `.gitignore`)
2. Proper file permissions are set in production
3. Regular backups are encrypted
4. Compliance with data protection regulations (GDPR, HIPAA, etc.)

## Limitations

- File-based storage has limitations for very high-volume scenarios
- No built-in encryption (consider filesystem-level encryption)
- Memory files grow over time (implement rotation/archival as needed)

## Future Enhancements

Potential improvements:

- Memory compression/archival for old sessions
- Full-text search capabilities
- Memory summarization for long-term users
- Export to other formats (JSON, PDF)
- Analytics dashboard for therapists
- Automatic pattern detection across sessions
