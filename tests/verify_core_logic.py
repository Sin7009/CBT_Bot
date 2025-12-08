import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent
from src.schemas import PatientState, TherapistDraft, SupervisorCritique, DistortionType

# Mock the client structure: client.chat.completions.create
# The CBTAgent uses `instructor.from_openai(AsyncOpenAI(...))` which wraps the client.
# We will mock the `client` attribute of the agent directly to avoid mocking `instructor`.

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client

@pytest.fixture
def agent(mock_client):
    # Initialize agent with dummy key and models
    agent = CBTAgent(api_key="dummy", model_therapist="gpt-4o", model_supervisor="gpt-4o")
    # Replace the internal client with our mock
    agent.client = mock_client
    return agent

@pytest.mark.asyncio
async def test_safety_risk_trigger(agent, mock_client):
    """
    Test that if the Left Brain detects a safety risk, the loop breaks immediately
    and returns a canned safety message.
    """
    # 1. Setup Mock for 'PatientState' (Safety Risk = True)
    unsafe_state = PatientState(
        current_emotion="Despair",
        intensity=10,
        distortion=DistortionType.CATASTROPHIZING,
        safety_risk=True
    )

    # Configure the mock to return unsafe_state for the first call (Supervisor Analysis)
    mock_client.chat.completions.create.side_effect = [unsafe_state]

    response = await agent.run("I want to end it all", [])

    assert "103" in response or "телефон доверия" in response
    # Ensure we didn't proceed to generate drafts
    assert mock_client.chat.completions.create.call_count == 1

@pytest.mark.asyncio
async def test_grounding_loop_success_first_try(agent, mock_client):
    """
    Test normal flow: Analysis -> Draft -> Critique (Approved) -> Return Draft
    """
    # 1. Analysis (Safe)
    safe_state = PatientState(
        current_emotion="Anxiety",
        intensity=6,
        distortion=DistortionType.CATASTROPHIZING,
        safety_risk=False
    )

    # 2. Draft (Therapist)
    draft = TherapistDraft(
        content="It sounds like you are worried. What is the evidence for that?",
        technique_used="Socratic Questioning"
    )

    # 3. Critique (Supervisor - Approved)
    critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=True,
        feedback="Good job."
    )

    mock_client.chat.completions.create.side_effect = [safe_state, draft, critique]

    response = await agent.run("I will fail everything", [])

    assert response == draft.content
    assert mock_client.chat.completions.create.call_count == 3

@pytest.mark.asyncio
async def test_grounding_loop_correction(agent, mock_client):
    """
    Test correction flow:
    Analysis -> Draft 1 (Bad) -> Critique 1 (Reject) -> Draft 2 (Good) -> Critique 2 (Approve)
    """
    # 1. Analysis
    safe_state = PatientState(
        current_emotion="Sadness",
        intensity=5,
        distortion=DistortionType.ALL_OR_NOTHING,
        safety_risk=False
    )

    # 2. Draft 1 (Bad - Advice)
    bad_draft = TherapistDraft(
        content="You should just go for a walk.",
        technique_used="Advice"
    )

    # 3. Critique 1 (Reject)
    bad_critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=False,
        feedback="Do not give advice. Ask questions."
    )

    # 4. Draft 2 (Good)
    good_draft = TherapistDraft(
        content="What makes you feel that way?",
        technique_used="Socratic Questioning"
    )

    # 5. Critique 2 (Approve)
    good_critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=True,
        feedback="Much better."
    )

    mock_client.chat.completions.create.side_effect = [
        safe_state,
        bad_draft,
        bad_critique,
        good_draft,
        good_critique
    ]

    response = await agent.run("I feel stuck.", [])

    assert response == good_draft.content
    assert mock_client.chat.completions.create.call_count == 5
