from .base import Q_COLS, QUANTILES, ForecastModel, qcol


def _chronos2(**kw):
    from .chronos2 import Chronos2Model
    return Chronos2Model(**kw)


# key in config -> factory(**params). Add TimesFM, SeasonalNaive, ETS ... here.
MODEL_REGISTRY = {
    "chronos2": _chronos2,
}


def get_model(name: str, **params) -> ForecastModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"ไม่รู้จักโมเดล {name!r}; มี {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


__all__ = ["ForecastModel", "QUANTILES", "Q_COLS", "qcol", "get_model", "MODEL_REGISTRY"]
