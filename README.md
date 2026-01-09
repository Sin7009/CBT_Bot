# Neuro-symbolic CBT Bot

A Telegram bot implementing a Cognitive Behavioral Therapy (CBT) agent using a neuro-symbolic "Two Hemispheres" architecture. It combines an empathetic "Therapist" model with a critical "Supervisor" model to ensure safe, protocol-adherent, and effective therapeutic responses.

## Features

*   **Two-Hemisphere Architecture:**
    *   **Therapist (Right Brain):** Generates empathetic, Socratic responses.
    *   **Supervisor (Left Brain):** Validates responses against CBT protocols and safety guidelines.
*   **Grounding Loop:** Iteratively refines responses based on Supervisor critique (up to 3 attempts).
*   **Safety First:** Detects crisis situations and provides emergency resources immediately.
*   **State Analysis:** Analyzes patient state (Automatic Thoughts, Intermediate Beliefs, Core Beliefs) before responding.
*   **Telegram Interface:** Real-time status updates ("Thinking...", "Consulting supervisor...") and typing indicators.
*   **Redis History:** Maintains conversation context (last 10 messages) for continuity.
*   **Markdown Memory Storage:** Persistent storage of therapeutic sessions in human-readable markdown files with rich metadata (see [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)).

## Prerequisites

*   Python 3.11+
*   Redis (for conversation history)
*   OpenAI API Key (via OpenRouter)

## Configuration

Copy `.env.example` to `.env` and configure the following variables:

```bash
cp .env.example .env
```

| Variable | Description | Default (in `src/config.py`) |
| :--- | :--- | :--- |
| `TELEGRAM_TOKEN` | Your Telegram Bot Token (from @BotFather). | *Required* |
| `OPENAI_API_KEY` | Your OpenRouter API Key. | *Required* |
| `REDIS_URL` | Redis connection string. | `redis://redis:6379/0` |
| `MODEL_THERAPIST` | Model for generation. | `google/gemini-2.5-flash` |
| `MODEL_SUPERVISOR` | Model for validation. | `deepseek/deepseek-v3.2-speciale` |
| `MEMORY_DIR` | Directory for memory storage. | `agent_memory` |
| `USE_MEMORY_STORAGE` | Enable markdown memory storage. | `true` |

> **Note on Models:** The default models (`gemini-2.5-flash` and `deepseek-v3.2-speciale`) are configured in the code. Ensure these models are available via your OpenRouter provider, or override them in your `.env` file with available alternatives (e.g., `google/gemini-pro`, `deepseek/deepseek-chat`).

## Installation & Usage

### Option A: Docker (Recommended)

1.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This will start both the Bot and the Redis container.

### Option B: Local Development

1.  **Start Redis:**
    Ensure a Redis instance is running locally or accessible via `REDIS_URL`.

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the bot:**
    ```bash
    python bot.py
    ```

## Development & Testing

To run the test suite (which verifies the agent's logic, safety checks, and edge-case handling):

1.  **Install test dependencies:** (Included in `requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run tests using pytest:**
    ```bash
    # Set PYTHONPATH to include the current directory
    export PYTHONPATH=$PYTHONPATH:.
    python -m pytest tests/
    ```

## Architecture Details

The bot uses `aiogram` for Telegram interactions and `instructor` for structured LLM outputs.

1.  **User Message:** Received via Telegram.
2.  **Analysis:** The Supervisor model analyzes the user's cognitive distortions and risk level.
3.  **Drafting:** The Therapist model generates a draft response.
4.  **Critique:** The Supervisor evaluates the draft.
    *   If **Approved**: Sent to the user.
    *   If **Rejected**: Feedback is fed back to the Therapist for regeneration (Grounding Loop).
5.  **Response:** The final validated response is sent to the user.
