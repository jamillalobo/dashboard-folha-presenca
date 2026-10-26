"""Índices (imagens/GT/predições), parsing dos JSONs e métricas por imagem."""
import csv
import io
import json
import re
from pathlib import Path

import numpy as np

import drive

MODELS = {"Nemotron": "nemotron", "Qwen": "qwen", "Ministral": "ministral"}
PROMPTS = {"Chain of Thought": "cot", "GEPA": "gepa", "Zero-shot": "zs"}
RESOLUTIONS = ["R1344", "R1792"]
N_LESSONS = 30


def _stem(name: str) -> str:
    return Path(name).stem


def gt_index() -> dict[str, dict[str, str]]:
    """image_id -> {'image': file_id, 'label': file_id} (ignora 'images reviewed')."""
    idx: dict[str, dict[str, str]] = {}
    for root in drive.GT_ROOTS:
        for fid, name in drive.list_folder(root):
            kind = {"images": "image", "labels": "label", "label": "label"}.get(name.lower())
            if not kind:
                continue
            for file_id, fname in drive.list_folder(fid):
                idx.setdefault(_stem(fname), {})[kind] = file_id
    return {k: v for k, v in idx.items() if len(v) == 2}


def _pred_folder_key(name: str) -> tuple[str, str] | None:
    """'images1_cot_nemotron' / 'images3_qwen_cot' -> (modelo, prompt)."""
    if not name.startswith("images") or "audit" in name:
        return None
    model = next((m for m in MODELS.values() if m in name), None)
    prompt = "cot" if "cot" in name else "gepa" if "gepa" in name else \
        "zs" if ("zero_shot" in name or name.endswith("_zs")) else None
    return (model, prompt) if model and prompt else None


def pred_index(model: str, prompt: str) -> dict[tuple[str, str], str]:
    """(image_id, resolução) -> file_id, juntando todas as pastas do modelo/prompt."""
    idx = {}
    for fid, name in drive.list_folder(drive.PRED_ROOT):
        if _pred_folder_key(name) != (MODELS[model], PROMPTS[prompt]):
            continue
        for sub_id, _ in drive.list_folder(fid):  # subpasta com o nome do modelo
            for file_id, fname in drive.list_folder(sub_id):
                m = re.match(r"(.+)__(R\d+)\.json$", fname)
                if m:
                    idx[(m.group(1), m.group(2))] = file_id
    return idx


def _matrix(students: list[dict]) -> np.ndarray:
    rows = []
    for s in students:
        lab = np.array(s["labels"], dtype=float).flatten()
        if lab.size != N_LESSONS:
            raise ValueError("labels com tamanho diferente de 30")
        rows.append(lab.astype(int))
    return np.array(rows)


def _parse_json(raw: bytes) -> dict:
    """JSON do arquivo, tolerando cercas markdown (```json ... ```) ao redor."""
    text = raw.decode("utf-8-sig")
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def parse_gt(raw: bytes) -> tuple[list[str], np.ndarray]:
    d = _parse_json(raw)
    if "students" not in d:  # GT vem embrulhado numa chave com o id da ficha
        d = next(iter(d.values()))
    return [s["student_name"] for s in d["students"]], _matrix(d["students"])


def _find_students(d: dict) -> list[dict]:
    """Acha a lista de alunos: direto, embrulhada numa chave de id, ou dentro de `raw_response`."""
    if "students" in d:
        return d["students"]
    if isinstance(d.get("raw_response"), str):
        return _find_students(_parse_json(d["raw_response"].encode()))
    if len(d) == 1 and isinstance(next(iter(d.values())), dict):
        return _find_students(next(iter(d.values())))
    raise KeyError("sem a chave 'students'")


def parse_pred(raw: bytes) -> tuple[np.ndarray | None, str]:
    """(matriz, '') da predição, ou (None, motivo) se o JSON do modelo for inválido."""
    try:
        return _matrix(_find_students(_parse_json(raw))), ""
    except (ValueError, KeyError, TypeError) as e:
        return None, f"{type(e).__name__}: {e}"


def confusion(gt: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return dict(TP=int(((gt == 1) & (pred == 1)).sum()), FP=int(((gt == 0) & (pred == 1)).sum()),
                FN=int(((gt == 1) & (pred == 0)).sum()), TN=int(((gt == 0) & (pred == 0)).sum()))


def scores(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn
    div = lambda a, b: a / b if b else float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n**2
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(accuracy=po, kappa=div(po - pe, 1 - pe), precision=prec, recall=rec,
                f1=div(2 * prec * rec, prec + rec), recall_absent=div(tn, tn + fp))


def _num(s: str) -> float:
    s = s.replace("%", "").replace("+", "").strip()
    return float(s.replace(".", "").replace(",", ".")) if s else float("nan")


def summary_rows(model: str, resolution: str, prompt: str) -> list[dict]:
    """Linhas da planilha consolidada (sem processamento, sem corte), uma por amostra."""
    rows = list(csv.reader(io.StringIO(drive.sheet_csv(drive.METRICS_SHEET))))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Amostra")
    cols = rows[head]
    out = []
    for r in rows[head + 1:]:
        d = dict(zip(cols, r))
        if (d.get("Tipo de processamento") == "Sem processamento" and d.get("Corte") == "Sem corte"
                and d.get("Modelo") == model and d.get("Resolução") == resolution
                and d.get("Tipo de prompt") == prompt):
            out.append(dict(
                amostra=d["Amostra"], folhas=int(_num(d["Folhas avaliadas"])),
                kappa=_num(d["Kappa oficial (micro)"]), json_invalidos=int(_num(d["JSON inválidos"])),
                gt_presencas=int(_num(d["Presenças GT"])), TP=int(_num(d["Presenças acertadas (TP)"])),
                TN=int(_num(d["Faltas acertadas (TN)"])), FP=int(_num(d["Falsas presenças (FP)"])),
                FN=int(_num(d["Presenças não detectadas (FN)"]))))
    return out
