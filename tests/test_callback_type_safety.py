import pytest
from unittest.mock import MagicMock, AsyncMock
from src.agent import CBTAgent

# Explanation:
# This test targets a "Type Consistency" failure in `CBTAgent.run`.
# The method signature defines `on_status_update` as `Optional[Callable[[str], None]]`.
# This type hint implies a synchronous function (one that returns None, not Awaitable[None]).
# However, the implementation uses `await on_status_update(...)`.
# If a developer (or another part of the system) passes a regular synchronous function matching the type hint,
# the `await` expression will raise a `TypeError` because the result is `None` (not awaitable),
# causing the agent to crash unexpectedly.
# This represents a contract violation where the implementation demands stricter types than the interface declares.

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock the create method to return a dummy object with safety_risk=False
    # so we can proceed to the point where callback is used in the loop if needed
    mock_response = MagicMock()
    mock_response.safety_risk = False
    mock_response.content = "Draft content"

    # We need to support multiple calls:
    # 1. Supervisor Analysis -> returns state with safety_risk=False
    # 2. Therapist Draft -> returns draft
    # 3. Supervisor Critique -> returns critique

    async def side_effect(*args, **kwargs):
        # Determine what kind of model is being called or just return generic mocks
        # For this test, we just need it to not crash on API calls
        return mock_response

    client.chat.completions.create = AsyncMock(side_effect=side_effect)
    return client

@pytest.fixture
def agent(mock_client):
    agent = CBTAgent(api_key="dummy", model_therapist="gpt-4o", model_supervisor="gpt-4o")
    agent.client = mock_client
    return agent

@pytest.mark.asyncio
async def test_run_with_synchronous_callback_crash(agent):
    """
    Test that passing a synchronous function as `on_status_update` causes a TypeError.
    The implementation awaits the callback result, so a sync function returning None
    will cause `await None`, raising TypeError.
    """

    # A synchronous callback function (matches Callable[[str], None])
    def sync_callback(status: str):
        pass

    # We expect this to crash with TypeError: object NoneType can't be used in 'await' expression
    # or similar message depending on python version
    with pytest.raises(TypeError):
        await agent.run(user_message="Hello", history=[], on_status_update=sync_callback)
