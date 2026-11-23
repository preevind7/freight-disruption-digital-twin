from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


class LoginRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"username": "dispatcher", "password": "Dispatch@123"}})

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class UserProfile(BaseModel):
    username: str
    name: str
    role: str = Field(..., description="dispatcher / operator / planner / researcher")
    role_title: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    user: UserProfile


class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"name": "Ravi Kumar", "username": "ravi", "password": "Ravi@2026", "role": "operator"}}
    )

    name: str = Field(..., min_length=2, max_length=60, description="Full name shown in the app")
    username: str = Field(..., description="3-32 characters: letters, digits, dot, underscore or hyphen")
    password: str = Field(..., max_length=128, description="At least 8 characters with at least one letter and one number")
    role: Literal["dispatcher", "operator", "planner", "researcher"]

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("name must be at least 2 characters")
        return value

    @field_validator("username")
    @classmethod
    def check_username(cls, value: str) -> str:
        value = value.strip().lower()
        if not USERNAME_PATTERN.match(value):
            raise ValueError("username must be 3-32 characters using letters, numbers, dot, underscore or hyphen (no spaces)")
        return value

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        if not re.search(r"[A-Za-z]", value) or not re.search(r"[0-9]", value):
            raise ValueError("password must contain at least one letter and one number")
        return value
