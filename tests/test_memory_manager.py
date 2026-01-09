"""
Tests for the memory manager module.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.memory_manager import MemoryManager, MemoryEntry


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_manager(temp_memory_dir):
    """Create a memory manager with a temporary directory."""
    return MemoryManager(temp_memory_dir)


@pytest.fixture
def sample_entry():
    """Create a sample memory entry."""
    return MemoryEntry(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        user_id="test_user_123",
        user_message="I feel anxious about my presentation",
        agent_response="What evidence do you have that supports this worry?",
        emotion="anxiety",
        intensity=7,
        thought_level="Автоматическая мысль (АМ)",
        primary_distortion="Катастрофизация",
        technique_used="Socratic Questioning"
    )


@pytest.mark.asyncio
async def test_save_memory_creates_file(memory_manager, sample_entry):
    """Test that saving memory creates a file."""
    await memory_manager.save_memory(sample_entry)
    
    file_path = memory_manager._get_user_file_path(sample_entry.user_id)
    assert file_path.exists()


@pytest.mark.asyncio
async def test_save_memory_content_format(memory_manager, sample_entry):
    """Test that saved memory has correct markdown format."""
    await memory_manager.save_memory(sample_entry)
    
    file_path = memory_manager._get_user_file_path(sample_entry.user_id)
    content = file_path.read_text(encoding="utf-8")
    
    # Check for key sections
    assert "# Memory Log for User" in content
    assert "## Session:" in content
    assert "### 🧠 Analysis" in content
    assert "### 💬 Conversation" in content
    assert "**User**:" in content
    assert "**Agent**:" in content
    assert sample_entry.user_message in content
    assert sample_entry.agent_response in content


@pytest.mark.asyncio
async def test_save_multiple_entries(memory_manager, sample_entry):
    """Test saving multiple entries appends to the file."""
    # Save first entry
    await memory_manager.save_memory(sample_entry)
    
    # Create and save second entry
    second_entry = MemoryEntry(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        user_id=sample_entry.user_id,
        user_message="What if I fail?",
        agent_response="What evidence suggests you will fail?",
        emotion="fear",
        intensity=8,
        thought_level="Автоматическая мысль (АМ)",
        primary_distortion="Катастрофизация",
        technique_used="Evidence Analysis"
    )
    await memory_manager.save_memory(second_entry)
    
    file_path = memory_manager._get_user_file_path(sample_entry.user_id)
    content = file_path.read_text(encoding="utf-8")
    
    # Both entries should be present
    assert content.count("## Session:") == 2
    assert sample_entry.user_message in content
    assert second_entry.user_message in content


@pytest.mark.asyncio
async def test_load_history_returns_correct_format(memory_manager, sample_entry):
    """Test that loading history returns messages in correct format."""
    await memory_manager.save_memory(sample_entry)
    
    history = await memory_manager.load_history(sample_entry.user_id)
    
    assert isinstance(history, list)
    assert len(history) == 2  # user message + agent response
    
    # Check first message (user)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == sample_entry.user_message
    
    # Check second message (agent)
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == sample_entry.agent_response


@pytest.mark.asyncio
async def test_load_history_respects_limit(memory_manager, sample_entry):
    """Test that load_history respects the limit parameter."""
    # Save 3 entries
    for i in range(3):
        entry = MemoryEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id=sample_entry.user_id,
            user_message=f"Message {i}",
            agent_response=f"Response {i}",
            emotion="test",
            intensity=5
        )
        await memory_manager.save_memory(entry)
    
    # Load with limit=2 (should return 4 messages: 2 user + 2 agent)
    history = await memory_manager.load_history(sample_entry.user_id, limit=2)
    
    assert len(history) <= 4  # At most 2 conversations (4 messages)


@pytest.mark.asyncio
async def test_load_history_nonexistent_user(memory_manager):
    """Test loading history for a user that doesn't exist."""
    history = await memory_manager.load_history("nonexistent_user")
    
    assert history == []


@pytest.mark.asyncio
async def test_get_user_stats(memory_manager, sample_entry):
    """Test getting user statistics."""
    # Before saving
    stats = await memory_manager.get_user_stats(sample_entry.user_id)
    assert stats["total_sessions"] == 0
    assert not stats["file_exists"]
    
    # After saving
    await memory_manager.save_memory(sample_entry)
    stats = await memory_manager.get_user_stats(sample_entry.user_id)
    assert stats["total_sessions"] == 1
    assert stats["file_exists"]


@pytest.mark.asyncio
async def test_clear_user_memory(memory_manager, sample_entry):
    """Test clearing user memory."""
    # Save entry
    await memory_manager.save_memory(sample_entry)
    
    file_path = memory_manager._get_user_file_path(sample_entry.user_id)
    assert file_path.exists()
    
    # Clear memory
    result = await memory_manager.clear_user_memory(sample_entry.user_id)
    assert result is True
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_list_users(memory_manager):
    """Test listing all users with stored memories."""
    # Create entries for multiple users
    user_ids = ["user_1", "user_2", "user_3"]
    
    for user_id in user_ids:
        entry = MemoryEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id=user_id,
            user_message="Test message",
            agent_response="Test response",
            emotion="test",
            intensity=5
        )
        await memory_manager.save_memory(entry)
    
    users = memory_manager.list_users()
    assert len(users) == 3
    assert set(users) == set(user_ids)


@pytest.mark.asyncio
async def test_concurrent_writes(memory_manager, sample_entry):
    """Test that concurrent writes are handled correctly."""
    # Create multiple entries for the same user
    async def save_entry(i):
        entry = MemoryEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id=sample_entry.user_id,
            user_message=f"Concurrent message {i}",
            agent_response=f"Concurrent response {i}",
            emotion="test",
            intensity=5
        )
        await memory_manager.save_memory(entry)
    
    # Run multiple saves concurrently
    await asyncio.gather(*[save_entry(i) for i in range(5)])
    
    # Verify all entries were saved
    file_path = memory_manager._get_user_file_path(sample_entry.user_id)
    content = file_path.read_text(encoding="utf-8")
    
    assert content.count("## Session:") == 5
    for i in range(5):
        assert f"Concurrent message {i}" in content


@pytest.mark.asyncio
async def test_entry_without_optional_fields(memory_manager):
    """Test saving an entry without optional fields."""
    minimal_entry = MemoryEntry(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        user_id="minimal_user",
        user_message="Minimal message",
        agent_response="Minimal response"
    )
    
    await memory_manager.save_memory(minimal_entry)
    
    file_path = memory_manager._get_user_file_path(minimal_entry.user_id)
    assert file_path.exists()
    
    # Load and verify
    history = await memory_manager.load_history(minimal_entry.user_id)
    assert len(history) == 2
    assert history[0]["content"] == "Minimal message"
