import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent
from src.schemas import PatientState, TherapistDraft, SupervisorCritique, DistortionType, ThoughtLevel

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
        thought_level=ThoughtLevel.CORE_BELIEF,
        primary_distortion=DistortionType.CATASTROPHIZING,
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
    # 1. Analysis (Safe, Intermediate Belief)
    safe_state = PatientState(
        current_emotion="Anxiety",
        intensity=6,
        thought_level=ThoughtLevel.INTERMEDIATE_BELIEF,
        primary_distortion=DistortionType.CATASTROPHIZING,
        safety_risk=False
    )

    # 2. Draft (Therapist)
    draft = TherapistDraft(
        content="It sounds like you have a rule that failing means disaster. Does it always?",
        technique_used="Consequence Analysis",
        target_level=ThoughtLevel.INTERMEDIATE_BELIEF
    )

    # 3. Critique (Supervisor - Approved)
    critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=True,
        correct_level_identification=True,
        feedback="Good job detecting the rule."
    )

    mock_client.chat.completions.create.side_effect = [safe_state, draft, critique]

    response = await agent.run("If I fail, my life is over.", [])

    assert response == draft.content
    assert mock_client.chat.completions.create.call_count == 3

@pytest.mark.asyncio
async def test_grounding_loop_correction_wrong_level(agent, mock_client):
    """
    Test correction flow:
    Analysis (Core Belief) -> Draft 1 (Treating as Automatic Thought) -> Critique 1 (Reject) -> Draft 2 (Core Belief Work) -> Critique 2 (Approve)
    """
    # 1. Analysis - Detected Core Belief ("I am unlovable")
    state = PatientState(
        current_emotion="Sadness",
        intensity=8,
        thought_level=ThoughtLevel.CORE_BELIEF,
        primary_distortion=DistortionType.LABELING,
        safety_risk=False
    )

    # 2. Draft 1 (Bad - Treating as AT)
    bad_draft = TherapistDraft(
        content="Did anyone say you are unlovable today?",
        technique_used="Evidence Check",
        target_level=ThoughtLevel.AUTOMATIC_THOUGHT
    )

    # 3. Critique 1 (Reject - Wrong Level)
    bad_critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=True, # Technique might be valid for AT, but level is wrong
        correct_level_identification=False,
        feedback="Client voiced a Core Belief ('I am unlovable'). Do not treat it as a situational thought. Use Downward Arrow."
    )

    # 4. Draft 2 (Good - Downward Arrow)
    good_draft = TherapistDraft(
        content="If it were true that you are unlovable, what would be the worst part of that for you?",
        technique_used="Downward Arrow",
        target_level=ThoughtLevel.CORE_BELIEF
    )

    # 5. Critique 2 (Approve)
    good_critique = SupervisorCritique(
        is_safe=True,
        adherence_to_protocol=True,
        correct_level_identification=True,
        feedback="Excellent application of Downward Arrow."
    )

    mock_client.chat.completions.create.side_effect = [
        state,
        bad_draft,
        bad_critique,
        good_draft,
        good_critique
    ]

    response = await agent.run("I am just completely unlovable.", [])

    assert response == good_draft.content
    assert mock_client.chat.completions.create.call_count == 5
