from pydantic import BaseModel, Field

# Modelo de entrada para la API
class PerfilPersonalidad(BaseModel):
    Extroversión: float = Field(..., ge=1.0, le=5.0)
    Responsabilidad: float = Field(..., ge=1.0, le=5.0)
    Amabilidad: float = Field(..., ge=1.0, le=5.0)
    Neuroticismo: float = Field(..., ge=1.0, le=5.0)
    Apertura: float = Field(..., ge=1.0, le=5.0)