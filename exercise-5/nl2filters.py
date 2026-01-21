# exercise-5/nl2filters.py
from __future__ import annotations
import json
import re
from typing import Optional, Dict, Any, List, Literal, Union
from pydantic import BaseModel, Field, field_validator

Number = Union[int, float]

class RangeOp(BaseModel):
    lt: Optional[Number] = Field(None, alias="$lt")
    gt: Optional[Number] = Field(None, alias="$gt")
    between: Optional[List[Number]] = Field(None, alias="$between")

    @field_validator("between")
    @classmethod
    def validate_between(cls, v: Optional[List[Number]]) -> Optional[List[Number]]:
        if v is None: return v
        return [min(v), max(v)]

    def to_chroma_filter(self) -> Dict[str, Any]:
        if self.between:
            return {"$and": [{"$gt": self.between[0]}, {"$lt": self.between[1]}]}
        out = {}
        if self.lt is not None: out["$lt"] = self.lt
        if self.gt is not None: out["$gt"] = self.gt
        return out

class NLFilter(BaseModel):
    genre: Optional[Union[str, Dict[str, List[str]]]] = None
    key: Optional[int] = None
    mode: Optional[int] = None
    popularity: Optional[RangeOp] = None
    energy: Optional[RangeOp] = None
    tempo: Optional[RangeOp] = None
    limit: Optional[int] = 5  # Defaulting to 5 if not found

def build_user_prompt(user_request: str) -> str:
    """Enhanced prompt to ensure the limit is extracted correctly."""
    return f"""Convert this music request to a JSON filter. 
User request: "{user_request}"

STRICT RULES:
1. If the user mentions a number (e.g., '15 songs', '7 tracks'), set the "limit" field to that EXACT number.
2. If the user says 'all', set "limit" to 100.
3. Use the schema: {{"genre": string, "limit": int, "popularity": {{"$gt": int}}}}
4. Return ONLY the JSON object. No prose."""

def interpret_nl_to_filter(raw_llm_json: str) -> NLFilter:
    try:
        match = re.search(r"\{.*\}", raw_llm_json, re.DOTALL)
        if match:
            raw_llm_json = match.group(0)
        data = json.loads(raw_llm_json)
        return NLFilter.model_validate(data)
    except:
        return NLFilter(limit=5)

NLFilter.model_rebuild()