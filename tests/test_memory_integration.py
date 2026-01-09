"""
Integration test to verify the memory system works with the agent.
"""

import pytest
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.agent import CBTAgent
from src.memory_manager import MemoryManager
from src.schemas import PatientState, TherapistDraft, SupervisorCritique, DistortionType, ThoughtLevel


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_manager(temp_memory_dir):
    """Create a memory manager with a temporary directory."""
    return MemoryManager(temp_memory_dir)


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def agent_with_memory(mock_client, memory_manager):
    """Create an agent with memory manager."""
    agent = CBTAgent(
        api_key="dummy",
        model_therapist="gpt-4o",
        model_supervisor="gpt-4o",
        memory_manager=memory_manager
    )
    agent.client = mock_client
    return agent


@pytest.mark.asyncio
async def test_agent_saves_interaction_to_memory(agent_with_memory, mock_client, memory_manager):
    """Test that agent successfully saves interactions to memory."""
    user_id = "test_user_123"
    user_message = "I'm feeling anxious about my presentation"
    
    # Mock responses
    safe_state = PatientState(
        current_emotion="anxiety",
        intensity=7,
        thought_level=ThoughtLevel.AUTOMATIC_THOUGHT,
        primary_distortion=DistortionType.CATASTROPHIZING,
        safety_risk=False
    )
    
    mock_draft = TherapistDraft(
        content="What evidence supports this worry?",
        technique_used="Socratic Questioning",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )
    
    mock_critique = SupervisorCritique(
        feedback="Good job",
        adherence_to_protocol=True,
        is_safe=True,
        correct_level_identification=True
    )
    
    def side_effect(*args, **kwargs):
        if kwargs.get('response_model') == PatientState:
            return safe_state
        elif kwargs.get('response_model') == TherapistDraft:
            return mock_draft
        elif kwargs.get('response_model') == SupervisorCritique:
            return mock_critique
        return safe_state
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    # Run the agent
    response = await agent_with_memory.run(user_message, [], user_id=user_id)
    
    # Verify response
    assert response == "What evidence supports this worry?"
    
    # Verify memory was saved
    stats = await memory_manager.get_user_stats(user_id)
    assert stats["total_sessions"] == 1
    assert stats["file_exists"]
    
    # Verify memory content
    history = await memory_manager.load_history(user_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == user_message
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == response


@pytest.mark.asyncio
async def test_agent_saves_safety_response_to_memory(agent_with_memory, mock_client, memory_manager):
    """Test that agent saves safety responses to memory."""
    user_id = "test_user_456"
    user_message = "I want to harm myself"
    
    # Mock state with safety risk
    unsafe_state = PatientState(
        current_emotion="despair",
        intensity=10,
        thought_level=ThoughtLevel.CORE_BELIEF,
        primary_distortion=DistortionType.CATASTROPHIZING,
        safety_risk=True
    )
    
    mock_client.chat.completions.create.side_effect = lambda *args, **kwargs: unsafe_state
    
    # Run the agent
    response = await agent_with_memory.run(user_message, [], user_id=user_id)
    
    # Verify safety response
    assert "скорую" in response or "телефон доверия" in response
    
    # Verify memory was saved
    stats = await memory_manager.get_user_stats(user_id)
    assert stats["total_sessions"] == 1
    
    # Verify memory content
    history = await memory_manager.load_history(user_id)
    assert len(history) == 2
    assert history[0]["content"] == user_message


@pytest.mark.asyncio
async def test_agent_continues_conversation_with_memory(agent_with_memory, mock_client, memory_manager):
    """Test that agent can continue conversations using stored memory."""
    user_id = "test_user_789"
    
    # First interaction
    user_message_1 = "I'm worried about failing"
    
    safe_state = PatientState(
        current_emotion="anxiety",
        intensity=6,
        thought_level=ThoughtLevel.AUTOMATIC_THOUGHT,
        primary_distortion=DistortionType.CATASTROPHIZING,
        safety_risk=False
    )
    
    mock_draft_1 = TherapistDraft(
        content="What makes you think you'll fail?",
        technique_used="Socratic Questioning",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )
    
    mock_critique = SupervisorCritique(
        feedback="Good",
        adherence_to_protocol=True,
        is_safe=True,
        correct_level_identification=True
    )
    
    def side_effect_1(*args, **kwargs):
        if kwargs.get('response_model') == PatientState:
            return safe_state
        elif kwargs.get('response_model') == TherapistDraft:
            return mock_draft_1
        elif kwargs.get('response_model') == SupervisorCritique:
            return mock_critique
        return safe_state
    
    mock_client.chat.completions.create.side_effect = side_effect_1
    
    response_1 = await agent_with_memory.run(user_message_1, [], user_id=user_id)
    assert response_1 == "What makes you think you'll fail?"
    
    # Second interaction - load history from memory
    user_message_2 = "I always fail at presentations"
    
    mock_draft_2 = TherapistDraft(
        content="Always? Can you think of any exceptions?",
        technique_used="Socratic Questioning",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )
    
    def side_effect_2(*args, **kwargs):
        if kwargs.get('response_model') == PatientState:
            return safe_state
        elif kwargs.get('response_model') == TherapistDraft:
            return mock_draft_2
        elif kwargs.get('response_model') == SupervisorCritique:
            return mock_critique
        return safe_state
    
    mock_client.chat.completions.create.side_effect = side_effect_2
    
    # Load history from memory
    history = await memory_manager.load_history(user_id)
    
    response_2 = await agent_with_memory.run(user_message_2, history, user_id=user_id)
    assert response_2 == "Always? Can you think of any exceptions?"
    
    # Verify both interactions are in memory
    stats = await memory_manager.get_user_stats(user_id)
    assert stats["total_sessions"] == 2


@pytest.mark.asyncio
async def test_agent_without_memory_manager_works(mock_client):
    """Test that agent works without memory manager (backward compatibility)."""
    agent = CBTAgent(
        api_key="dummy",
        model_therapist="gpt-4o",
        model_supervisor="gpt-4o",
        memory_manager=None
    )
    agent.client = mock_client
    
    user_message = "I'm stressed"
    
    safe_state = PatientState(
        current_emotion="stress",
        intensity=5,
        thought_level=ThoughtLevel.AUTOMATIC_THOUGHT,
        primary_distortion=DistortionType.NO_DISTORTION,
        safety_risk=False
    )
    
    mock_draft = TherapistDraft(
        content="Tell me more about your stress",
        technique_used="Open Question",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )
    
    mock_critique = SupervisorCritique(
        feedback="Good",
        adherence_to_protocol=True,
        is_safe=True,
        correct_level_identification=True
    )
    
    def side_effect(*args, **kwargs):
        if kwargs.get('response_model') == PatientState:
            return safe_state
        elif kwargs.get('response_model') == TherapistDraft:
            return mock_draft
        elif kwargs.get('response_model') == SupervisorCritique:
            return mock_critique
        return safe_state
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    # Run without user_id (should not crash)
    response = await agent.run(user_message, [])
    assert response == "Tell me more about your stress"
