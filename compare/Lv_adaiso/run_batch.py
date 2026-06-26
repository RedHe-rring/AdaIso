import subprocess
import time
import csv
import traceback
from pathlib import Path

import trimesh


# =========================
# 配置区
# =========================

EXE_PATH = r"E:\Remeshing\2024Lv_Adaptively-Isotropic-Remeshing-main\Adaptively-Isotropic-Remeshing-main\SHFVCG2021remesh_Hist\x64\Debug\SHFVCGRemesh_Hist.exe"

INPUT_DIR = r"E:\adaiso\cpp_adaiso\examples"

OUTPUT_WORK_DIR = r".\batch_run"

PREPROCESSING = 0          # 0: 不预处理, 1: 预处理
ISOTROPIC_JUDGE = 2        # 1: isotropic, 2: anisotropic
MESHSCALE = 1.2

RECURSIVE = True           # 是否递归处理子文件夹
CONVERT_OBJ_TOO = False    # True: obj 也重新导出一份; False: obj 直接送入 exe


# =========================
# 工具函数
# =========================

def load_mesh_as_single_mesh(mesh_path: Path):
    """
    读取 obj/off/ply，包括 binary ply。
    如果 trimesh 读出来是 Scene，则合并成一个 mesh。
    """
    loaded = trimesh.load(mesh_path, force=None, process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)

        if not meshes:
            raise ValueError(f"文件中没有可用 mesh: {mesh_path}")

        mesh = trimesh.util.concatenate(meshes)
        return mesh

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    raise TypeError(f"不支持的 mesh 类型: {type(loaded)}")


def convert_to_obj(input_path: Path, converted_dir: Path) -> Path:
    """
    将 off/ply/obj 转成 obj。
    对于 binary ply，trimesh 会自动读取并导出为 obj。
    """
    converted_dir.mkdir(parents=True, exist_ok=True)

    output_obj = converted_dir / f"{input_path.stem}.obj"

    mesh = load_mesh_as_single_mesh(input_path)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"模型没有顶点: {input_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"模型没有面: {input_path}")

    mesh.export(output_obj, file_type="obj")

    return output_obj


def collect_model_files(input_dir: Path, recursive=True):
    """
    收集 obj/off/ply 文件。
    """
    supported_exts = {".obj", ".off", ".ply"}

    if recursive:
        files = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
    else:
        files = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        ]

    return sorted(files)


def run_exe_for_one_model(
    exe_path: Path,
    obj_path: Path,
    preprocessing: int,
    isotropic_judge: int,
    meshscale: float,
    work_dir: Path,
    log_file: Path,
):
    """
    调用 exe 跑单个 obj 文件，并记录 stdout/stderr。
    """
    cmd = [
        str(exe_path),
        str(obj_path),
        str(preprocessing),
        str(isotropic_judge),
        str(meshscale),
    ]

    start_time = time.time()

    with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
        f.write("Command:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("Output:\n")

        process = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )

        output_lines = []

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            output_lines.append(line)

        process.wait()

    end_time = time.time()

    return {
        "return_code": process.returncode,
        "duration_sec": round(end_time - start_time, 4),
        "output": "".join(output_lines),
    }


# =========================
# 主批处理函数
# =========================

def batch_run():
    exe_path = Path(EXE_PATH).resolve()
    input_dir = Path(INPUT_DIR).resolve()
    work_dir = Path(OUTPUT_WORK_DIR).resolve()

    converted_dir = work_dir / "converted_obj"
    logs_dir = work_dir / "logs"
    csv_log_path = work_dir / "batch_log.csv"

    work_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not exe_path.exists():
        raise FileNotFoundError(f"找不到 exe 文件: {exe_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入文件夹: {input_dir}")

    model_files = collect_model_files(input_dir, recursive=RECURSIVE)

    print(f"找到 {len(model_files)} 个模型文件")

    records = []

    for index, model_path in enumerate(model_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(model_files)}] Processing: {model_path}")
        print("=" * 80)

        record = {
            "index": index,
            "input_file": str(model_path),
            "input_ext": model_path.suffix.lower(),
            "used_obj_file": "",
            "converted": False,
            "status": "",
            "return_code": "",
            "duration_sec": "",
            "log_file": "",
            "error": "",
        }

        try:
            suffix = model_path.suffix.lower()

            if suffix == ".obj" and not CONVERT_OBJ_TOO:
                used_obj_path = model_path
                converted = False
            else:
                used_obj_path = convert_to_obj(model_path, converted_dir)
                converted = True

            record["used_obj_file"] = str(used_obj_path)
            record["converted"] = converted

            log_file = logs_dir / f"{model_path.stem}.log"
            record["log_file"] = str(log_file)

            result = run_exe_for_one_model(
                exe_path=exe_path,
                obj_path=used_obj_path,
                preprocessing=PREPROCESSING,
                isotropic_judge=ISOTROPIC_JUDGE,
                meshscale=MESHSCALE,
                work_dir=work_dir,
                log_file=log_file,
            )

            record["return_code"] = result["return_code"]
            record["duration_sec"] = result["duration_sec"]

            if result["return_code"] == 0:
                record["status"] = "success"
                print(f"完成: {model_path.name}, 耗时 {result['duration_sec']} 秒")
            else:
                record["status"] = "failed"
                record["error"] = f"exe 返回非零状态码: {result['return_code']}"
                print(f"失败: {model_path.name}, return_code={result['return_code']}")

        except Exception as e:
            record["status"] = "error"
            record["error"] = repr(e)

            error_log_file = logs_dir / f"{model_path.stem}_python_error.log"
            record["log_file"] = str(error_log_file)

            with open(error_log_file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(traceback.format_exc())

            print(f"Python 处理错误: {model_path.name}")
            print(repr(e))

        records.append(record)

        # 每处理一个文件就刷新 csv，防止中途崩溃丢日志
        write_csv_log(csv_log_path, records)

    print("\n批处理完成")
    print(f"总日志文件: {csv_log_path}")


def write_csv_log(csv_log_path: Path, records):
    fieldnames = [
        "index",
        "input_file",
        "input_ext",
        "used_obj_file",
        "converted",
        "status",
        "return_code",
        "duration_sec",
        "log_file",
        "error",
    ]

    with open(csv_log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    batch_run()