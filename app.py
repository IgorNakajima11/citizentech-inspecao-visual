from pathlib import Path

import streamlit as st
from PIL import Image

from inspecao import modelo

RAIZ = Path(__file__).resolve().parent
DEMO = RAIZ / "demonstracao"

st.set_page_config(page_title="CitizenTech - Inspeção por imagem", layout="wide")
pacote, _ = st.cache_resource(show_spinner="Carregando o modelo...")(modelo.carregar)()

st.title("CitizenTech: inspeção de fissuras por imagem")
st.write("Envie a foto de uma superfície de concreto (parede, calçada, viaduto). O sistema aponta onde há sinal de "
         "fissura e com que confiança, para ajudar a Prefeitura a decidir **onde mandar uma inspeção**.")
st.warning("A presença de fissura na imagem, sozinha, **não indica a gravidade** do dano nem a segurança da estrutura. "
           "O resultado serve só para priorizar a visita de um técnico.")

with st.sidebar:
    limiar = st.slider("Limiar de confiança para marcar fissura", 0.05, 0.95, float(pacote["limiar"]), 0.05,
                       help=f"Padrão {pacote['limiar']}, escolhido na validação para o melhor F1. Mais baixo: pega mais "
                            "fissuras, com mais alarmes falsos. Mais alto: menos alarmes, mas pode deixar passar fissuras.")
    st.caption("Modelo: ResNet50 pré-treinado no ImageNet (congelado) + regressão logística treinada em 1.811 "
               "imagens da base Concrete Crack Images.")

aba_img, aba_video = st.tabs(["Analisar imagem", "Monitoramento por vídeo"])

with aba_img:
    exemplos = sorted(DEMO.glob("*.jpg"))
    c1, c2 = st.columns([2, 1])
    enviada = c1.file_uploader("Foto da superfície (jpg ou png)", type=["jpg", "jpeg", "png"])
    exemplo = c2.selectbox("Ou use um exemplo", ["(nenhum)"] + [p.name for p in exemplos])
    img = None
    if enviada is not None:
        img = Image.open(enviada)
    elif exemplo != "(nenhum)":
        img = Image.open(DEMO / exemplo)
    if img is not None:
        with st.spinner("Analisando..."):
            r = modelo.analisar(img, limiar)
        col_a, col_b = st.columns(2)
        col_a.image(img, caption="Imagem enviada", width="stretch")
        col_b.image(r.imagem_marcada, caption="Regiões marcadas (vermelho = possível fissura)", width="stretch")
        if r.tem_fissura:
            st.error(f"Possível fissura encontrada. Confiança máxima: {r.confianca:.0%}. Sugestão: incluir o local na "
                     f"lista de inspeção.")
        else:
            st.success(f"Nenhuma fissura acima do limiar. Maior confiança de fissura: {r.confianca:.0%}.")
        marcados = sum(b["fissura"] for b in r.blocos)
        st.caption(f"{len(r.blocos)} região(ões) analisada(s), {marcados} marcada(s). Tempo: {r.tempo_ms:.0f} ms.")

with aba_video:
    st.write("Simula o monitoramento urbano: cada quadro do vídeo é analisado e marcado. Use o vídeo de exemplo "
             "(sequência de 40 imagens da base de teste) ou envie um vídeo curto.")
    video = st.file_uploader("Vídeo curto (mp4)", type=["mp4", "mov", "webm"])
    usar_exemplo = st.button("Processar o vídeo de exemplo")
    dados = video.getvalue() if video is not None else ((DEMO / "monitoramento_simulado.mp4").read_bytes() if usar_exemplo else None)
    if dados:
        with st.spinner("Processando quadro a quadro..."):
            r = modelo.processar_video(dados, limiar)
        st.video(r["video"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Quadros analisados", len(r["quadros"]))
        c2.metric("Quadros com possível fissura", r["quadros_com_fissura"])
        c3.metric("Velocidade", f"{r['fps']} quadros/s", help=f"{r['tempo_medio_ms']} ms por quadro, só CPU")
        st.dataframe(r["quadros"], hide_index=True)
