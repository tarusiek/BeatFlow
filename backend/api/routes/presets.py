from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["presets"])

PRESETS = {
    "default": {
        "label": "Default",
        "description": "Adaptive — parameters chosen by the decision engine based on your vocal.",
    },
    "hip_hop": {
        "label": "Hip-Hop",
        "description": "Aggressive compression, forward presence boost at 4 kHz, punch-forward low-mid.",
    },
    "rnb": {
        "label": "R&B",
        "description": "Smooth compression, warm presence at 3.5 kHz, subtle saturation, wider reverb.",
    },
    "pop": {
        "label": "Pop",
        "description": "Balanced chain, bright air at 5 kHz, punchy attack, clean and wide.",
    },
}


@router.get("/presets")
def list_presets():
    return PRESETS
