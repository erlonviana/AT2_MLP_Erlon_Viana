from pydantic import BaseModel, Field


class PredicaoInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Idade do cliente", examples=[45])
    job: str = Field(..., description="Profissão", examples=["admin."])
    marital: str = Field(..., description="Estado civil", examples=["married"])
    education: str = Field(..., description="Escolaridade", examples=["secondary"])
    default: str = Field(..., description="Tem crédito em default?", examples=["no"])
    balance: int = Field(..., ge=-10000, le=100000, description="Saldo médio anual")
    housing: str = Field(..., description="Tem financiamento habitacional?", examples=["yes"])
    loan: str = Field(..., description="Tem empréstimo pessoal?", examples=["no"])
    contact: str = Field(..., description="Tipo de contato", examples=["cellular"])
    day: int = Field(..., ge=1, le=31, description="Dia do último contato")
    month: str = Field(..., description="Mês do último contato", examples=["may"])
    duration: int = Field(..., ge=0, le=5000, description="Duração da ligação (s)")
    campaign: int = Field(..., ge=1, le=100, description="Nº de contatos nesta campanha")
    pdays: int = Field(..., ge=-1, le=1000, description="Dias desde o último contato (-1 = nunca)")
    previous: int = Field(..., ge=0, le=100, description="Nº de contatos anteriores")
    poutcome: str = Field(..., description="Resultado da campanha anterior", examples=["unknown"])


class PredicaoOutput(BaseModel):
    categoria_prevista: str
    score_confianca: float = Field(..., ge=0.0, le=1.0)