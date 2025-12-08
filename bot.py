import os
import asyncio
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from redis.asyncio import Redis

from src.agent import CBTAgent

# Конфиг из переменных окружения
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY") # Или OpenRouter
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Модели (можно менять на GPT-4o / Claude 3.5 Sonnet через OpenRouter)
MODEL_THERAPIST = os.getenv("MODEL_THERAPIST", "gpt-4o")
MODEL_SUPERVISOR = os.getenv("MODEL_SUPERVISOR", "gpt-4o")

bot = Bot(token=TOKEN)
dp = Dispatcher()
redis = Redis.from_url(REDIS_URL, decode_responses=True)
agent = CBTAgent(OPENAI_KEY, MODEL_THERAPIST, MODEL_SUPERVISOR)

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("Привет. Я твой КПТ-тренер. Расскажи, что тебя беспокоит?")
    # Очистка истории при старте
    await redis.delete(f"history:{message.from_user.id}")

@dp.message()
async def chat(message: types.Message):
    user_id = message.from_user.id
    user_text = message.text

    # Загружаем историю (последние 10 сообщений)
    try:
        raw_history = await redis.lrange(f"history:{user_id}", 0, -1)
        history = [json.loads(msg) for msg in raw_history][::-1] if raw_history else []
    except Exception as e:
        # Fallback if redis fails or is not present, though in prod it should be there.
        # This helps with local testing if one forgets redis.
        print(f"Redis error: {e}")
        history = []

    status_msg = await message.answer("Thinking... (Neuro-symbolic validation)")

    try:
        response = await agent.run(user_text, history)

        # Обновляем историю
        try:
            await redis.lpush(f"history:{user_id}", json.dumps({"role": "user", "content": user_text}))
            await redis.lpush(f"history:{user_id}", json.dumps({"role": "assistant", "content": response}))
            await redis.ltrim(f"history:{user_id}", 0, 10) # Храним только 10 последних
        except Exception as e:
            print(f"Redis write error: {e}")

        await status_msg.edit_text(response)
    except Exception as e:
        await status_msg.edit_text(f"Error: {str(e)}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
