"""US 참고 실행: 상위 폴더 strategy.py를 그대로 쓴다 (규칙 같음, 파라미터만 us/lab.yaml)."""
import importlib.util as _u
from pathlib import Path as _P

_spec = _u.spec_from_file_location("_parent_strategy", _P(__file__).resolve().parents[1] / "strategy.py")
_m = _u.module_from_spec(_spec)
_spec.loader.exec_module(_m)
generate_trades = _m.generate_trades
if hasattr(_m, "fit"):
    fit = _m.fit
