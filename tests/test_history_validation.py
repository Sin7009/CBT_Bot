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
    is gracefully handled by filtering out the invalid elements.
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

    # Mocks for generation loop
    from src.schemas import TherapistDraft, SupervisorCritique
    mock_draft = TherapistDraft(
        content="This is a therapeutic response.",
        technique_used="Socratic Questioning",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )
    mock_critique = SupervisorCritique(
        feedback="Good job",
        adherence_to_protocol=True,
        is_safe=True,
        correct_level_identification=True
    )

    # We expect the API client (mocked) to receive this garbage and likely fail if validation is missing.
    # To simulate the real world, we make the mock raise a TypeError or ValueError
    # when it encounters invalid messages, replicating strict API library behavior.
    def side_effect(*args, **kwargs):
        messages = kwargs.get('messages', [])
        # Iterate and check for validity logic similar to what httpx/openai would do
        for msg in messages:
            if not isinstance(msg, dict):
                raise TypeError(f"Invalid message type: {type(msg)}")

        # Helper to return the correct model based on response_model argument (inferred from call order or model arg)
        # 1. Analysis (PatientState)
        # 2. Draft (TherapistDraft)
        # 3. Critique (SupervisorCritique)
        if kwargs.get('response_model') == PatientState:
            return safe_state
        elif kwargs.get('response_model') == TherapistDraft:
            return mock_draft
        elif kwargs.get('response_model') == SupervisorCritique:
            return mock_critique
        return safe_state

    mock_client.chat.completions.create.side_effect = side_effect

    # The agent should NOT crash now. It should filter valid items and proceed.
    # We await the result to make sure it finishes.
    result = await agent.run(user_message, malformed_history)

    assert result == "This is a therapeutic response."

    # Verify that the invalid history was NOT passed to the client
    # The crash should NOT happen.

    # The generation call is the second call (index 1) or third depending on how many calls happened.
    # Calls: 1. Analysis, 2. Draft, 3. Critique.
    assert mock_client.chat.completions.create.call_count == 3

    # Check arguments of the Draft generation call (where history is used)
    # Analysis call (0) doesn't use history. Draft call (1) uses history.
    draft_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    messages = draft_call_kwargs['messages']

    # The history is injected after the system prompt (index 0)
    # The filtered history should contain ONLY the valid message.

    # "invalid_string_message" and None should be absent.
    assert "invalid_string_message" not in messages
    assert None not in messages

    # The valid message should be present
    assert any(m.get("content") == "Valid message" for m in messages if isinstance(m, dict))
