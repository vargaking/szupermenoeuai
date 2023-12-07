from tortoise import fields, models
from fastapi import HTTPException, status

from enum import Enum, StrEnum
import time
import datetime
from typing import Optional

from pydantic import BaseModel

from common.cancer_types import CancerType, get_cancer_type


class Death(models.Model):
    person_id: int
    death_date: datetime.date
    death_datetime: datetime.datetime
    death_type_concept_id: int
    cause_concept_id: int
    cause_source_value: str
    cause_source_concept_id: int