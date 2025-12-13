import pytest
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent
from src.schemas import PatientState, DistortionType, ThoughtLevel

# Explanation:
# This test targets the handling of malformed data in the 'history' list.
# The 'history' is typically loaded from Redis (deserialized from JSON).
# If the history contains elements that are not valid message dictionaries (e.g. strings, None, or dicts with missing keys),
# the `CBTAgent.run` method currently concatenates them blindly into the `messages` list.
# When this list is passed to the OpenAI/Instructor client, it causes a validation error or crash
# because the API expects a strict schema for messages.
# This represents a fragility in the agent's input processing: it relies entirely on the caller (bot.py)
# to provide a perfectly sanitized history, rather than validating its own inputs.

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
async def test_run_with_malformed_history_elements(agent, mock_client):
    """
    Test that passing a history list containing invalid types (e.g., strings, None)
    causes the agent to crash or fail when calling the API, as it performs no validation.
    """
    # Malformed history: contains a string and None, which are not valid message objects
    malformed_history = [
        {"role": "user", "content": "Valid message"},
        "invalid_string_message",
        None
    ]

    user_message = "Help me"

    # Create a mock state that passes the safety check
    safe_state = PatientState(
        current_emotion="sadness",
        intensity=5,
        thought_level=ThoughtLevel.AUTOMATIC_THOUGHT,
        primary_distortion=DistortionType.NO_DISTORTION,
        safety_risk=False
    )

    # We expect the API client (mocked) to receive this garbage and likely fail.
    # To simulate the real world, we make the mock raise a TypeError or ValueError
    # when it encounters invalid messages, replicating strict API library behavior.
    def side_effect(*args, **kwargs):
        messages = kwargs.get('messages', [])
        # Iterate and check for validity logic similar to what httpx/openai would do
        for msg in messages:
            if not isinstance(msg, dict):
                raise TypeError(f"Invalid message type: {type(msg)}")

        # If valid messages (like the first Analysis call), return a safe state
        # The agent expects a PatientState object (or similar Pydantic model) from the first call.
        # But wait, response_model is used. Instructor returns an instance of response_model.
        # We need to return the expected model instance depending on the call.
        # However, for this test, simply returning safe_state for the first call is enough.
        # The second call (Generation) won't even reach return because of the exception.
        return safe_state

    mock_client.chat.completions.create.side_effect = side_effect

    # The agent should crash because it doesn't validate history before using it
    with pytest.raises(TypeError, match="Invalid message type"):
        await agent.run(user_message, malformed_history)

    # Verify that the invalid history was indeed passed to the client
    # This confirms the "garbage in, garbage out" behavior of the function
    # The crash should happen on the SECOND call (TherapistDraft generation)
    # The first call (PatientState analysis) uses only user_message and should succeed.
    assert mock_client.chat.completions.create.call_count == 2

    # Check arguments of the last call (the failing one)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs['messages']

    # The history is injected after the system prompt (index 0)
    # So we expect the malformed elements to be present in the messages list
    assert "invalid_string_message" in messages
    assert None in messages
