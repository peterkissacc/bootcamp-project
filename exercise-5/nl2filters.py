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
    key: Optional[int] = None   # 0-11
    mode: Optional[int] = None  # 0: minor, 1: major
    explicit: Optional[bool] = None
    popularity: Optional[RangeOp] = None
    energy: Optional[RangeOp] = None
    danceability: Optional[RangeOp] = None
    valence: Optional[RangeOp] = None
    tempo: Optional[RangeOp] = None
    instrumentalness: Optional[RangeOp] = None
    acousticness: Optional[RangeOp] = None
    limit: Optional[int] = None
    sort_by: Optional[Literal["popularity_desc", "popularity_asc"]] = None

SYSTEM_PROMPT = """You are a Music Theory & Search Specialist. Convert requests to JSON.

KEY MAPPING: C=0, C#/Db=1, D=2, D#/Eb=3, E=4, F=5, F#/Gb=6, G=7, G#/Ab=8, A=9, A#/Bb=10, B=11
MODE MAPPING: minor=0, major=1

RULES:
- Multiple genres (e.g. "pop or rock") -> {{"genre": {{"$in": ["pop", "rock"]}}}}
- "fast" -> tempo > 140, "energetic" -> energy > 0.8, "chill" -> energy < 0.4
- If a specific key is mentioned (e.g. "C minor"), set "key" and "mode".

EXAMPLES:
User: "pop song in c minor"
JSON: {{"genre": "pop", "key": 0, "mode": 0, "limit": 5}}

STRICT RULE: Return ONLY JSON. Use double braces for all JSON structure.
Example: {{"limit": 1, "genre": "techno"}}"""

def interpret_nl_to_filter(raw_llm_json: str) -> NLFilter:
    try:
        match = re.search(r"\{.*\}", raw_llm_json, re.DOTALL)
        if match:
            raw_llm_json = match.group(0)
        data = json.loads(raw_llm_json)
        return NLFilter.model_validate(data)
    except:
        return NLFilter()

NLFilter.model_rebuild()