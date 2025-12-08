from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class DistortionType(str, Enum):
    NO_DISTORTION = "Нет искажений"
    ALL_OR_NOTHING = "Черно-белое мышление"
    CATASTROPHIZING = "Катастрофизация"
    OVERSIMPLIFICATION = "Сверхобобщение"
    MIND_READING = "Чтение мыслей"
    SHOULD_STATEMENTS = "Долженствование"
    LABELING = "Навешивание ярлыков"

class PatientState(BaseModel):
    """Анализ состояния пациента (Левое полушарие - Вход)."""
    current_emotion: str = Field(..., description="Названная или подразумеваемая эмоция.")
    intensity: int = Field(..., ge=1, le=10, description="Интенсивность эмоции 1-10.")
    distortion: DistortionType = Field(..., description="Тип когнитивного искажения.")
    safety_risk: bool = Field(False, description="True ЕСЛИ есть намеки на суицид или селф-харм.")

class TherapistDraft(BaseModel):
    """Черновик ответа Терапевта (Правое полушарие)."""
    content: str = Field(..., description="Текст ответа пациенту.")
    technique_used: str = Field(..., description="Название использованной техники КПТ (например, 'Сократовский диалог').")

class SupervisorCritique(BaseModel):
    """Вердикт Супервизора (Левое полушарие - Выход)."""
    is_safe: bool = Field(..., description="Безопасен ли ответ для пациента?")
    adherence_to_protocol: bool = Field(..., description="Соблюден ли протокол КПТ (валидация -> исследование -> нет советов)?")
    feedback: str = Field(..., description="Инструкция для Терапевта, что исправить.")
