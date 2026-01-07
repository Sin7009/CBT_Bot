import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent

# Explanation:
# This test verifies the fix for "Type Consistency" in `CBTAgent.run`.
# Previously, passing a synchronous function crashed because `await` was used on `None`.
# The fix ensures both sync and async callbacks work correctly.

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock the create method to return a dummy object with safety_risk=False
    # so we can proceed to the point where callback is used in the loop if needed
    mock_response = MagicMock()
    mock_response.safety_risk = False
    mock_response.content = "Draft content"
    mock_response.adherence_to_protocol = True
    mock_response.is_safe = True
    mock_response.correct_level_identification = True

    async def side_effect(*args, **kwargs):
        return mock_response

    client.chat.completions.create = AsyncMock(side_effect=side_effect)
    return client

@pytest.fixture
def agent(mock_client):
    agent = CBTAgent(api_key="dummy", model_therapist="gpt-4o", model_supervisor="gpt-4o")
    agent.client = mock_client
    return agent

@pytest.mark.asyncio
async def test_run_with_synchronous_callback_success(agent):
    """
    Test that passing a synchronous function as `on_status_update` works correctly.
    """
    callback_mock = MagicMock()

    def sync_callback(status: str):
        callback_mock(status)

    # Should not raise TypeError
    await agent.run(user_message="Hello", history=[], on_status_update=sync_callback)

    # Verify callback was called at least once (e.g. "Анализирую мысли...")
    assert callback_mock.call_count >= 1
    callback_mock.assert_any_call("Анализирую мысли...")

@pytest.mark.asyncio
async def test_run_with_asynchronous_callback_success(agent):
    """
    Test that passing an asynchronous function as `on_status_update` works correctly.
    """
    callback_mock = AsyncMock()

    async def async_callback(status: str):
        await callback_mock(status)

    # Should not raise TypeError
    await agent.run(user_message="Hello", history=[], on_status_update=async_callback)

    # Verify callback was called at least once
    assert callback_mock.call_count >= 1
    callback_mock.assert_any_call("Анализирую мысли...")
