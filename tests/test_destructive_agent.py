import pytest
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent

# Explanation:
# This test targets the handling of "None" input for the user_message.
# In a real scenario, this occurs when a user sends a non-text message (e.g., sticker, photo).
# The current logic in `CBTAgent.run` constructs a message dictionary: {"role": "user", "content": user_message}.
# If user_message is None, the dictionary becomes {"role": "user", "content": None}.
# The OpenAI API (and instructor) typically requires 'content' to be a string or a list of content parts.
# Passing None causes a BadRequestError or Validation Error at the library/API level.
# Since `CBTAgent.run` has no internal error handling for this specific validation failure,
# it will crash (raise an unhandled exception), which is then caught by the generic catch-all in bot.py,
# leading to a generic "Internal Error" response instead of a graceful handling or ignoring of the input.

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client

@pytest.fixture
def agent(mock_client):
    agent = CBTAgent(api_key="dummy", model_therapist="gpt-4o", model_supervisor="gpt-4o")
    agent.client = mock_client
    return agent

@pytest.mark.asyncio
async def test_agent_run_with_none_message(agent, mock_client):
    """
    Test that passing None as user_message (e.g. from a sticker) propagates
    invalid data to the API client, eventually causing a crash/error.
    """
    # Simulate the library raising an error when content is None
    # We use ValueError here to represent a validation error from 'instructor' or 'openai'
    def side_effect(*args, **kwargs):
        messages = kwargs.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'user' and msg.get('content') is None:
                raise ValueError("Content cannot be None for user role")
        return MagicMock()

    mock_client.chat.completions.create.side_effect = side_effect

    # We expect the function to crash because it lacks try/except for this case
    with pytest.raises(ValueError, match="Content cannot be None"):
        await agent.run(user_message=None, history=[])

    # Verify that the agent actually attempted to call the API with None content
    # (i.e. it didn't filter it out beforehand)
    mock_client.chat.completions.create.assert_called()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs['messages']

    # Check the messages structure passed to the first call (Analysis phase)
    user_msg = next((m for m in messages if m['role'] == 'user'), None)
    assert user_msg is not None
    assert user_msg['content'] is None
