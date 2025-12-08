import asyncio
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from redis.asyncio import Redis

from src.agent import CBTAgent
from src.config import settings

bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher()
redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
agent = CBTAgent(
    settings.OPENAI_API_KEY,
    settings.MODEL_THERAPIST,
    settings.MODEL_SUPERVISOR,
    base_url=settings.OPENAI_BASE_URL
)

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
    await bot.send_chat_action(chat_id=user_id, action="typing")

    async def update_status(text: str):
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass # Ignore edit errors (e.g. same text)

    try:
        response = await agent.run(user_text, history, on_status_update=update_status)

        # Обновляем историю
        try:
            await redis.lpush(f"history:{user_id}", json.dumps({"role": "user", "content": user_text}))
            await redis.lpush(f"history:{user_id}", json.dumps({"role": "assistant", "content": response}))
            await redis.ltrim(f"history:{user_id}", 0, 10) # Храним только 10 последних
        except Exception as e:
            print(f"Redis write error: {e}")

        await status_msg.edit_text(response)
    except Exception as e:
        import logging
        logging.error(f"Internal error processing message: {e}", exc_info=True)
        await status_msg.edit_text("Произошла внутренняя ошибка. Мы уже работаем над этим. Попробуйте нажать /start.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
