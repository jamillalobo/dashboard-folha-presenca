import argparse
import html
import io
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
.legend span{margin-right:14px;font-size:13px}
.legend .c{display:inline-block;vertical-align:-2px;margin-right:4px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Indexando Drive…")
def gt_index():
    return data.gt_index()


@st.cache_data(show_spinner="Indexando predições…")
def pred_index(model, prompt):
    return data.pred_index(model, prompt)


def grid(names, shown, gt=None):
    """Tabela de células. Sem `gt`, desenha GT; com `gt`, marca FP/FN contra o GT."""
    rows = []
    for i, name in enumerate(names):
        cells = []
        for j in range(data.N_LESSONS):
            v = shown[i, j] if i < len(shown) else 0
            cls = "p" if v else ""
            if gt is not None:
                g = gt[i, j]
                cls = "fp" if (v and not g) else "fn" if (g and not v) else cls
            cells.append(f'<td><span class="c {cls}"></span></td>')
        rows.append(f'<tr><td class="n">{html.escape(name)}</td>{"".join(cells)}</tr>')
    head = "".join(f'<td style="text-align:center;color:#888">{j + 1}</td>' for j in range(data.N_LESSONS))
    return f'<table class="grid"><tr><td></td>{head}</tr>{"".join(rows)}</table>'


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

# ---- imagem selecionada --------------------------------------------------
try:
    names, gt = data.parse_gt(gt_raw)
except (ValueError, KeyError, TypeError, StopIteration) as e:
    st.error(f"Ground truth inválido ou fora do formato esperado ({type(e).__name__}: {e}).")
    st.stop()
pred, pred_error = data.parse_pred(pred_raw)
pred_ok = pred is not None
if pred_ok and pred.shape != gt.shape:
    st.warning(f"Predição tem {pred.shape[0]} alunos e o GT {gt.shape[0]}; comparando as primeiras linhas.")
    r = min(pred.shape[0], gt.shape[0])
    names, gt, pred = names[:r], gt[:r], pred[:r]

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

st.markdown('<div class="legend"><span><i class="c p"></i>Presente (1)</span>'
            '<span><i class="c"></i>Ausente (0)</span><span><i class="c fp"></i>Falsa presença (FP)</span>'
            '<span><i class="c fn"></i>Presença perdida (FN)</span></div>', unsafe_allow_html=True)

left, mid, right = st.columns([1, 1.6, 1.6])
with left:
    st.subheader("Imagem")
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    st.image(img, width="stretch")
with mid:
    st.subheader("Ground truth")
    st.caption(f"{len(names)} alunos × {data.N_LESSONS} aulas")
    st.markdown(grid(names, gt), unsafe_allow_html=True)
with right:
    st.subheader("Predição do modelo")
    st.caption("Erros marcados contra os labels")
    if pred_ok:
        st.markdown(grid(names, pred, gt), unsafe_allow_html=True)
    else:
        st.caption("Sem predição válida para exibir.")
