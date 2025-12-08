from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# 1. Полный список искажений по А. Беку
class DistortionType(str, Enum):
    NO_DISTORTION = "Нет искажений"
    DICHOTOMOUS_THINKING = "Дихотомическое мышление (Ч/Б)"
    CATASTROPHIZING = "Катастрофизация"
    PAST_PREDICTION = "Апелляция к прошлому"
    DISCOUNTING_POSITIVE = "Обесценивание позитивного"
    EMOTIONAL_REASONING = "Эмоциональное обоснование"
    LABELING = "Навешивание ярлыков"
    MAGNIFICATION_MINIMIZATION = "Магнификация/Минимизация"
    MENTAL_FILTER = "Мысленный фильтр"
    MIND_READING = "Чтение мыслей"
    OVERGENERALIZATION = "Сверхгенерализация"
    PERSONALIZATION = "Персонализация"
    EXCESSIVE_RESPONSIBILITY = "Чрезмерная ответственность"
    SHOULD_STATEMENTS = "Долженствование"
    TUNNEL_VISION = "Туннельное мышление"

# 2. Уровни когниций (Новая сущность!)
class ThoughtLevel(str, Enum):
    AUTOMATIC_THOUGHT = "Автоматическая мысль (АМ)"
    INTERMEDIATE_BELIEF = "Промежуточное убеждение (Правило/Если...то)"
    CORE_BELIEF = "Глубинное убеждение (Я-концепция)"

# 3. Обновленное состояние пациента
class PatientState(BaseModel):
    """Снимок психического состояния клиента в моменте."""
    current_emotion: str = Field(..., description="Эмоция, которую испытывает клиент.")
    intensity: int = Field(..., ge=1, le=10, description="Интенсивность 1-10.")

    thought_level: ThoughtLevel = Field(
        ...,
        description="На каком уровне мыслит клиент? АМ (ситуативно), ПУ (правила жизни) или ГУ (суть личности)."
    )

    primary_distortion: DistortionType = Field(..., description="Доминирующее когнитивное искажение.")
    safety_risk: bool = Field(False, description="Есть риск суицида или селф-харма.")

# 4. Черновик терапевта (добавляем стратегию)
class TherapistDraft(BaseModel):
    content: str
    technique_used: str = Field(..., description="Например: 'Падающая стрела' для ГУ, 'Анализ доказательств' для АМ.")
    target_level: ThoughtLevel = Field(..., description="На каком уровне ведется работа.")

# 5. Критика супервизора
class SupervisorCritique(BaseModel):
    is_safe: bool
    adherence_to_protocol: bool
    correct_level_identification: bool = Field(..., description="Правильно ли терапевт определил уровень (АМ/ПУ/ГУ)?")
    feedback: str
