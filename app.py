import argparse
import html
import io
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import data
import drive

st.set_page_config(page_title="Dashboard SGP", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1rem}
h1{padding-top:0}
.grid{border-collapse:separate;border-spacing:2px;font-size:11px}
.grid td{padding:0}.grid .n{padding-right:8px;white-space:nowrap;max-width:150px;overflow:hidden;
text-overflow:ellipsis;color:#444}
.c{display:block;width:11px;height:11px;border-radius:2px;border:1.5px solid #cfcfcf;box-sizing:border-box}
.c.p{background:#2b2b2b;border-color:#2b2b2b}.c.fp{background:#c8501a;border-color:#c8501a}
.c.fn{border-color:#c8501a;background:#fff}
.c.na{border-style:dashed;border-color:#8a8a8a;background:transparent}
.c.ex{border-color:#2f6fd0;background:#fff}
.c.fp.nd{background-image:repeating-linear-gradient(45deg,#fff 0 1.5px,transparent 1.5px 4px)}
.c.fn.nd{border-style:dashed}
.legend span{margin-right:14px;font-size:13px}
.legend .c{display:inline-block;vertical-align:-2px;margin-right:4px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Indexando Drive…")
def gt_index():
    return data.gt_index()


@st.cache_data(show_spinner="Indexando predições…")
def pred_index(model, prompt):
    return data.pred_index(model, prompt)


CELL_CLASS = {data.TP: "p", data.TN: "", data.FP: "fp", data.FN: "fn", data.FP_NOPRED: "fp nd",
              data.FN_NOPRED: "fn nd", data.FP_EXTRA: "ex", data.OUT: "na", data.NONE: None}


def _table(names, cell_classes, width):
    """Tabela HTML; classe None = célula inexistente. Colunas além do GT têm o número em azul."""
    rows = []
    for name, classes in zip(names, cell_classes):
        cells = "".join("<td></td>" if c is None else f'<td><span class="c {c}"></span></td>' for c in classes)
        rows.append(f'<tr><td class="n">{html.escape(name)}</td>{cells}</tr>')
    head = "".join(f'<td style="text-align:center;color:{"#2f6fd0" if j >= data.N_LESSONS else "#888"}">{j + 1}</td>'
                   for j in range(width))
    return f'<table class="grid"><tr><td></td>{head}</tr>{"".join(rows)}</table>'


def grid_gt(names, gt):
    return _table(names, [["na" if np.isnan(v) else "p" if v else "" for v in row] for row in gt], gt.shape[1])


def grid_pred(names, codes):
    """Grade da predição colorida pela classe de cada célula (`data.classify`)."""
    extra = [f"Aluno a mais {k + 1}" for k in range(codes.shape[0] - len(names))]
    return _table(names + extra, [[CELL_CLASS[c] for c in row] for row in codes], codes.shape[1])


def structure_note(n_gt, pred) -> str:
    """Diferenças de estrutura entre predição e GT (alunos e aulas), ou '' se batem."""
    parts = []
    if pred.shape[0] > n_gt:
        parts.append(f"{pred.shape[0] - n_gt} aluno(s) a mais")
    if pred.shape[0] < n_gt:
        parts.append(f"{n_gt - pred.shape[0]} aluno(s) omitido(s)")
    days = sorted(set(data.returned_days(pred[:n_gt])) - {data.N_LESSONS})
    if days:
        parts.append(f"{' e '.join(map(str, days))} aulas de {data.N_LESSONS}")
    return ", ".join(parts[:-1]) + " e " + parts[-1] if len(parts) > 1 else "".join(parts)


def _source() -> str:
    """Fonte dos dados, escolhida na linha de comando: streamlit run app.py -- --source drive|upload"""
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["drive", "upload"], default="drive")
    return p.parse_known_args(sys.argv[1:])[0].source


SOURCE = _source()

st.title("Visualização da Folha de Presença")

if SOURCE == "drive":
    # O dropdown de imagens fica no topo, mas depende dos filtros abaixo dele:
    # reservo o espaço agora e preencho depois de ler os filtros.
    image_box = st.container()

    # ---- filtros -------------------------------------------------------------
    f_model, f_res, f_prompt, f_reload = st.columns([1, 1, 1, 0.8], vertical_alignment="bottom")
    model = f_model.selectbox("Modelo", list(data.MODELS))
    resolution = f_res.selectbox("Resolução", data.RESOLUTIONS)
    prompt = f_prompt.selectbox("Prompt", list(data.PROMPTS))
    if f_reload.button("Recarregar do Drive"):
        drive.clear_cache()
        st.cache_data.clear()
        st.rerun()

    try:
        gts, preds = gt_index(), pred_index(model, prompt)
    except Exception as e:  # rede/Drive fora do ar, ou .env incompleto
        st.error(f"Não consegui ler o Drive: {e}")
        st.stop()

    available = sorted((i for i in gts if (i, resolution) in preds),
                       key=lambda s: int(s.split("_")[0][3:]))
    if not available:
        st.warning("Nenhuma imagem com predição para esse modelo/prompt/resolução.")
        st.stop()
    with image_box:
        image_id = st.selectbox("Imagem", available,
                                format_func=lambda s: f"{s}  ({available.index(s) + 1}/{len(available)})")

    # ---- métricas gerais (planilha consolidada) ------------------------------
    with st.expander(f"Métricas gerais · {model} · {resolution} · {prompt} (sem processamento, sem corte)", expanded=True):
        srows = data.summary_rows(model, resolution, prompt)
        if not srows:
            st.info("Sem linhas na planilha consolidada para essa combinação.")
        else:
            df = pd.DataFrame(srows)
            tot = {k: int(df[k].sum()) for k in ("TP", "TN", "FP", "FN")}
            sc = data.scores(tot)
            c = st.columns(5)
            c[0].metric("Kappa (agregado)", f"{sc['kappa']:.3f}")
            c[1].metric("Presenças acertadas (TP)", f"{tot['TP']:,}".replace(",", "."),
                        f"de {int(df.gt_presencas.sum()):,}".replace(",", "."), delta_color="off")
            c[2].metric("Faltas acertadas (TN)", f"{tot['TN']:,}".replace(",", "."))
            c[3].metric("Acurácia", f"{sc['accuracy']:.1%}")
            c[4].metric("JSON inválidos", f"{int(df.json_invalidos.sum())} / {int(df.folhas.sum())}")
            st.dataframe(df.rename(columns=dict(kappa="Kappa (planilha)")), hide_index=True, width="stretch")

    image_bytes = drive.download(gts[image_id]["image"])
    gt_raw = drive.download(gts[image_id]["label"])
    pred_raw = drive.download(preds[(image_id, resolution)])
    pred_info = f"Resolução {resolution}"
else:
    # ---- upload: arraste os três arquivos ------------------------------------
    u_img, u_gt, u_pred = st.columns(3)
    up_img = u_img.file_uploader("Imagem", type=["jpg", "jpeg", "png"])
    up_gt = u_gt.file_uploader("Ground truth (JSON)", type=["json", "md", "txt"])
    up_pred = u_pred.file_uploader("Predição (JSON)", type=["json", "md", "txt"])
    if not (up_img and up_gt and up_pred):
        st.info("Arraste a imagem, o ground truth e a predição para visualizar.")
        st.stop()
    image_bytes, gt_raw, pred_raw = up_img.getvalue(), up_gt.getvalue(), up_pred.getvalue()
    # no upload o modelo não vem no arquivo; a resolução sai do nome (img…__R1344.json)
    m = re.search(r"__(R\d+)\.\w+$", up_pred.name)
    pred_info = f"Resolução {m.group(1)}" if m else ""

# ---- imagem selecionada --------------------------------------------------
try:
    names, gt = data.parse_gt(gt_raw)
except (ValueError, KeyError, TypeError, StopIteration) as e:
    st.error(f"Ground truth inválido ou fora do formato esperado ({type(e).__name__}: {e}).")
    st.stop()
pred, pred_error = data.parse_pred(pred_raw)
pred_ok = pred is not None
codes = data.classify(gt, pred) if pred_ok else None

if pred_ok:
    conf = data.confusion(gt, pred)
    s = data.scores(conf)
    cols = st.columns(8)
    for col, (label, v) in zip(cols, [("Acurácia", s["accuracy"]), ("Kappa", s["kappa"]),
                                      ("Precisão", s["precision"]), ("Recall", s["recall"]),
                                      ("F1", s["f1"]), ("Recall (ausente)", s["recall_absent"])]):
        col.metric(label, f"{v:.3f}")
    cols[6].metric("TP / FP", f"{conf['TP']} / {conf['FP']}")
    cols[7].metric("FN / TN", f"{conf['FN']} / {conf['TN']}")
else:
    st.error(f"A predição está inválida ou fora do formato esperado ({pred_error}).")

LEGEND = [("p", "Presente (1)"), ("", "Ausente (0)"), ("fp", "Falsa presença (FP)"), ("fn", "Presença perdida (FN)"),
          ("fp nd", "Null/ausente com GT 0 (FP)"), ("fn nd", "Null/ausente com GT 1 (FN)"),
          ("ex", "Aula ou aluno a mais (FP)"), ("na", "Fora da conta (null)")]
st.markdown('<div class="legend">' + "".join(f'<span><i class="c {cls}"></i>{text}</span>' for cls, text in LEGEND)
            + "</div>", unsafe_allow_html=True)

left, mid, right = st.columns([1, 1.6, 1.6])
with left:
    st.subheader("Imagem")
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    st.image(img, width="stretch")
with mid:
    st.markdown(f'### Ground truth <span style="font-size:.9rem;font-weight:400;color:#777">'
                f'{len(names)} alunos × {data.N_LESSONS} aulas</span>', unsafe_allow_html=True)
    st.markdown(grid_gt(names, gt), unsafe_allow_html=True)
with right:
    sub = f' <span style="font-size:.9rem;font-weight:400;color:#777">{pred_info}</span>' if pred_info else ""
    st.markdown(f"### Predição do modelo{sub}", unsafe_allow_html=True)
    if pred_ok:
        if note := structure_note(len(names), pred):
            st.warning(f"Estrutura diferente do GT: {note}.")
        st.markdown(grid_pred(names, codes), unsafe_allow_html=True)
    else:
        st.caption("Sem predição válida para exibir.")
