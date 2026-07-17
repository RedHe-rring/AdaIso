import importlib.util
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "generate_v5_0_0_sqrt_vs_linear_experiment.py"

spec = importlib.util.spec_from_file_location("sqrt_vs_linear_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


base.MODELS = [
    p.name
    for p in sorted(base.EXAMPLES.glob("T*.off"), key=natural_key)
    if re.fullmatch(r"T\d+\.off", p.name)
]
base.RESULT_ROOT = base.ROOT / "results" / "v5_0_0_T_series_sqrt_vs_linear"
base.REPORT_PATH = base.ROOT / "修改计划与实验结果" / "v5.0.0 T系列 sqrt_vs_linear实验报告.md"


if __name__ == "__main__":
    print("T-series models:")
    for name in base.MODELS:
        print(" ", name)
    base.main()
