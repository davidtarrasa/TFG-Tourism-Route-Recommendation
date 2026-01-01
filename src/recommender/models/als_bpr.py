"""
CF implícito (opcional):
- ALS o BPR sobre matriz usuario-POI de check-ins.
- Requiere librería como `implicit`.
"""


def train_implicit_model(interactions):
    # TODO: entrenar ALS/BPR si se incluye dependencia.
    raise NotImplementedError


def score_cf(model, user_id, candidates):
    # TODO: puntuar candidatos con el modelo entrenado.
    raise NotImplementedError
