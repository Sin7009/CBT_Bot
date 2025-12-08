import asyncio
from openai import AsyncOpenAI
import instructor
from .schemas import PatientState, TherapistDraft, SupervisorCritique
from .prompts import THERAPIST_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT

class CBTAgent:
    def __init__(self, api_key: str, model_therapist: str, model_supervisor: str):
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key))
        self.model_therapist = model_therapist
        self.model_supervisor = model_supervisor

    async def run(self, user_message: str, history: list) -> str:
        # 1. АНАЛИЗ СОСТОЯНИЯ (Левое полушарие)
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
        internal_log = [] # Для отладки (можно выводить в консоль)

        for i in range(3): # Максимум 3 попытки
            # Генерация черновика
            draft = await self.client.chat.completions.create(
                model=self.model_therapist,
                response_model=TherapistDraft,
                messages=messages + [{"role": "system", "content": f"Предыдущие замечания супервизора: {internal_log}"}] if internal_log else messages
            )

            # Критика
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

            internal_log.append(critique.feedback)
            print(f"⚠️ Попытка {i+1} отклонена: {critique.feedback}")

        return "Извини, я затрудняюсь сформулировать терапевтический ответ прямо сейчас. Попробуй перефразировать."
