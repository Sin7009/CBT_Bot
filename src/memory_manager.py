"""
Markdown-based memory storage system for CBT agent.
Stores conversation history and agent analysis in human-readable markdown files.
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from filelock import FileLock


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the agent's history."""
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


class MemoryManager:
    """
    Manages agent memory storage using markdown files.
    Each user has their own markdown file with conversation history.
    """
    
    def __init__(self, memory_dir: str = "agent_memory"):
        """
        Initialize the memory manager.
        
        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_user_file_path(self, user_id: str) -> Path:
        """Get the path to a user's memory file."""
        return self.memory_dir / f"user_{user_id}.md"
    
    def _get_lock_path(self, user_id: str) -> Path:
        """Get the path to a user's lock file."""
        return self.memory_dir / f"user_{user_id}.lock"
    
    async def save_memory(self, entry: MemoryEntry) -> None:
        """
        Save a memory entry to the user's markdown file.
        
        Args:
            entry: MemoryEntry to save
        """
        file_path = self._get_user_file_path(entry.user_id)
        lock_path = self._get_lock_path(entry.user_id)
        
        # Use file lock for concurrent access
        lock = FileLock(str(lock_path), timeout=10)
        
        try:
            # Run file operations in thread pool to avoid blocking
            await asyncio.to_thread(self._save_with_lock, lock, file_path, entry)
        except Exception as e:
            print(f"Error saving memory: {e}")
            raise
    
    def _save_with_lock(self, lock: FileLock, file_path: Path, entry: MemoryEntry) -> None:
        """Helper method to save memory with file lock (runs in thread pool)."""
        with lock:
            # Read existing content
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
            else:
                content = f"# Memory Log for User {entry.user_id}\n\n"
                content += f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                content += "---\n\n"
            
            # Append new entry
            content += self._format_entry(entry)
            
            # Write back
            file_path.write_text(content, encoding="utf-8")
    
    def _format_entry(self, entry: MemoryEntry) -> str:
        """
        Format a memory entry as markdown.
        
        Args:
            entry: MemoryEntry to format
            
        Returns:
            Formatted markdown string
        """
        md = f"## Session: {entry.timestamp}\n\n"
        
        # Add analysis if available
        if (entry.emotion or entry.intensity or entry.thought_level or 
            entry.primary_distortion or entry.analysis):
            md += "### 🧠 Analysis\n\n"
            if entry.emotion:
                md += f"- **Emotion**: {entry.emotion}\n"
            if entry.intensity:
                md += f"- **Intensity**: {entry.intensity}/10\n"
            if entry.thought_level:
                md += f"- **Thought Level**: {entry.thought_level}\n"
            if entry.primary_distortion:
                md += f"- **Primary Distortion**: {entry.primary_distortion}\n"
            md += "\n"
        
        # Add conversation
        md += "### 💬 Conversation\n\n"
        md += f"**User**: {entry.user_message}\n\n"
        md += f"**Agent**: {entry.agent_response}\n\n"
        
        # Add technique if available
        if entry.technique_used:
            md += f"**Technique Used**: {entry.technique_used}\n\n"
        
        md += "---\n\n"
        
        return md
    
    async def load_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Load conversation history for a user.
        
        Args:
            user_id: User ID to load history for
            limit: Maximum number of entries to load
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        file_path = self._get_user_file_path(user_id)
        
        if not file_path.exists():
            return []
        
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            return self._parse_history(content, limit)
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def _parse_history(self, content: str, limit: int) -> List[Dict[str, str]]:
        """
        Parse markdown content to extract conversation history.
        
        Args:
            content: Markdown content
            limit: Maximum number of entries to extract
            
        Returns:
            List of message dictionaries
        """
        messages = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for user message
            if line.startswith('**User**:'):
                user_msg = line.replace('**User**:', '').strip()
                
                # Look for agent response (should be nearby)
                j = i + 1
                agent_msg = None
                while j < len(lines) and j < i + 10:
                    if lines[j].strip().startswith('**Agent**:'):
                        agent_msg = lines[j].strip().replace('**Agent**:', '').strip()
                        break
                    j += 1
                
                # Add messages in order (user first, then agent)
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if agent_msg:
                    messages.append({"role": "assistant", "content": agent_msg})
                
                # Stop if we've reached the limit
                if len(messages) >= limit * 2:
                    break
            
            i += 1
        
        # Return most recent messages (latest entries are at the end)
        # Take the last 'limit' conversations (limit * 2 messages)
        return messages[-(limit * 2):]
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user's memory.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with statistics
        """
        file_path = self._get_user_file_path(user_id)
        
        if not file_path.exists():
            return {
                "total_sessions": 0,
                "file_exists": False
            }
        
        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        session_count = content.count("## Session:")
        
        return {
            "total_sessions": session_count,
            "file_exists": True,
            "file_path": str(file_path)
        }
    
    async def clear_user_memory(self, user_id: str) -> bool:
        """
        Clear all memory for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_user_file_path(user_id)
        lock_path = self._get_lock_path(user_id)
        
        try:
            if file_path.exists():
                lock = FileLock(str(lock_path), timeout=10)
                with lock:
                    await asyncio.to_thread(file_path.unlink)
            return True
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
    
    def list_users(self) -> List[str]:
        """
        List all users with stored memories.
        
        Returns:
            List of user IDs
        """
        user_files = self.memory_dir.glob("user_*.md")
        users = []
        for f in user_files:
            # Remove the "user_" prefix from filename
            # stem gives us "user_<user_id>", so we remove first 5 chars ("user_")
            user_id = f.stem[5:]  # Skip "user_" prefix
            users.append(user_id)
        return users
