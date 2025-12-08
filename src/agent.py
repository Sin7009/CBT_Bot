import asyncio
from typing import Callable, Optional
from openai import AsyncOpenAI
import instructor
from .schemas import PatientState, TherapistDraft, SupervisorCritique
from .prompts import THERAPIST_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT

class CBTAgent:
    def __init__(self, api_key: str, model_therapist: str, model_supervisor: str, base_url: str = None):
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
        self.model_therapist = model_therapist
        self.model_supervisor = model_supervisor

    async def run(self, user_message: str, history: list, on_status_update: Optional[Callable[[str], None]] = None) -> str:
        # 1. АНАЛИЗ СОСТОЯНИЯ (Левое полушарие)
        if on_status_update:
            await on_status_update("Анализирую мысли...")

        state = await self.client.chat.completions.create(
            model=self.model_supervisor,
            response_model=PatientState,
            messages=[
                {"role": "system", "content": "Проанализируй состояние пациента. Будь бдителен к рискам."},
                {"role": "user", "content": user_message}
            ]
        )

        # SAFETY VALVE (Предохранитель)
        if state.safety_risk:
            return "Я ИИ-ассистент и не могу помочь в кризисной ситуации. Пожалуйста, позвоните в скорую (103) или телефон доверия."

        # 2. ЦИКЛ ГЕНЕРАЦИИ (Grounding Loop)
        messages = [{"role": "system", "content": THERAPIST_SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]
        last_draft = None
        last_critique = None

        for i in range(3): # Максимум 3 попытки
            if on_status_update:
                await on_status_update(f"Формулирую ответ (попытка {i+1})...")

            # Подготовка контекста ретрая
            current_messages = list(messages)
            if last_draft and last_critique:
                retry_context = {
                    "role": "system",
                    "content": (
                        f"Твой предыдущий вариант ответа: '{last_draft.content}'\n"
                        f"Был отклонен супервизором по причине: '{last_critique.feedback}'\n"
                        "Попробуй снова, исправив эти ошибки."
                    )
                }
                current_messages.append(retry_context)

            # Генерация черновика
            draft = await self.client.chat.completions.create(
                model=self.model_therapist,
                response_model=TherapistDraft,
                messages=current_messages
            )

            # Критика
            if on_status_update:
                await on_status_update("Консультируюсь с супервизором...")

            critique = await self.client.chat.completions.create(
                model=self.model_supervisor,
                response_model=SupervisorCritique,
                messages=[
                    {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Сообщение пациента: {user_message}\n\nОтвет терапевта: {draft.content}"}
                ]
            )

            if critique.adherence_to_protocol and critique.is_safe and critique.correct_level_identification:
                return draft.content

            last_draft = draft
            last_critique = critique
            print(f"⚠️ Попытка {i+1} отклонена: {critique.feedback}")

        return "Извини, я затрудняюсь сформулировать терапевтический ответ прямо сейчас. Попробуй перефразировать."
