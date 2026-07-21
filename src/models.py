"""Pydantic models for function definitions, parameters, and results."""

from typing import Any, Dict, List
from pydantic import BaseModel, field_validator


class ParameterSchema(BaseModel):
    """Schema for a single function parameter."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure type is one of the supported JSON schema types."""
        allowed = {"number", "string", "boolean", "integer"}
        if v not in allowed:
            raise ValueError(f"Unsupported parameter type: {v}. Must be one of {allowed}")
        return v


class FunctionDefinition(BaseModel):
    """Schema for a single function definition."""

    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: ParameterSchema

    def get_param_names(self) -> List[str]:
        """Return list of parameter names in insertion order."""
        return list(self.parameters.keys())

    def get_param_type(self, param_name: str) -> str:
        """Return the type string for a given parameter name."""
        return self.parameters[param_name].type


class FunctionCallResult(BaseModel):
    """Schema for a single output entry."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
