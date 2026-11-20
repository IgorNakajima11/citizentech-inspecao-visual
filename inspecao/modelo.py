import io
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from fastembed import ImageEmbedding
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parents[1]
MODELO_PADRAO = RAIZ / "modelo" / "classificador_resnet50.pkl"
LADO = 227


@dataclass
class Resultado:
    tem_fissura: bool
    confianca: float
    blocos: list
    imagem_marcada: Image.Image
    tempo_ms: float


@lru_cache(maxsize=2)
def carregar(caminho: str = str(MODELO_PADRAO)):
    with open(caminho, "rb") as f:
        pacote = pickle.load(f)
    return pacote, ImageEmbedding(pacote["modelo_embedding"])


def _probabilidades(blocos: list[Image.Image]) -> np.ndarray:
    pacote, emb = carregar()
    vetores = np.array(list(emb.embed(blocos, batch_size=16)), dtype=np.float32)
    return pacote["classificador"].predict_proba(vetores)[:, 1]


def _grade(img: Image.Image) -> list[tuple[int, int, int, int]]:
    w, h = img.size
    if w <= LADO * 1.5 and h <= LADO * 1.5:
        return [(0, 0, w, h)]
    nx, ny = max(1, round(w / LADO)), max(1, round(h / LADO))
    return [(int(i * w / nx), int(j * h / ny), int((i + 1) * w / nx), int((j + 1) * h / ny))
            for j in range(ny) for i in range(nx)]


def analisar(img: Image.Image, limiar: float | None = None) -> Resultado:
    pacote, _ = carregar()
    limiar = pacote["limiar"] if limiar is None else limiar
    img = img.convert("RGB")
    t0 = time.perf_counter()
    caixas = _grade(img)
    probs = _probabilidades([img.crop(c) for c in caixas])
    tempo = (time.perf_counter() - t0) * 1000
    marcada = img.copy()
    d = ImageDraw.Draw(marcada)
    blocos = []
    for c, p in zip(caixas, probs):
        achou = bool(p >= limiar)
        blocos.append({"caixa": c, "confianca_fissura": float(p), "fissura": achou})
        if achou:
            d.rectangle(c, outline=(220, 30, 30), width=max(2, img.size[0] // 150))
            d.rectangle((c[0], c[1], c[0] + 62, c[1] + 16), fill=(220, 30, 30))
            d.text((c[0] + 3, c[1] + 2), f"{p:.0%}", fill="white")
    maior = float(max(probs))
    return Resultado(maior >= limiar, maior, blocos, marcada, tempo)


def processar_video(dados: bytes, limiar: float | None = None, max_quadros: int = 150, a_cada: int = 1) -> dict:
    entrada = av.open(io.BytesIO(dados))
    fluxo = entrada.streams.video[0]
    saida_buf = io.BytesIO()
    saida = av.open(saida_buf, mode="w", format="mp4")
    quadros, linhas, t_total = 0, [], 0.0
    out = None
    for i, quadro in enumerate(entrada.decode(fluxo)):
        if quadros >= max_quadros:
            break
        if i % a_cada:
            continue
        img = quadro.to_image()
        r = analisar(img, limiar)
        t_total += r.tempo_ms
        if out is None:
            out = saida.add_stream("libx264", rate=max(1, int(round(float(fluxo.average_rate or 5) / a_cada))))
            out.width, out.height = (img.size[0] // 2) * 2, (img.size[1] // 2) * 2
            out.pix_fmt = "yuv420p"
        marcada = r.imagem_marcada.crop((0, 0, out.width, out.height))
        d = ImageDraw.Draw(marcada)
        rotulo = f"{'INSPECIONAR' if r.tem_fissura else 'ok'}  fissura {r.confianca:.0%}"
        d.rectangle((0, out.height - 18, out.width, out.height), fill=(0, 0, 0))
        d.text((4, out.height - 15), rotulo, fill=(255, 80, 80) if r.tem_fissura else (120, 255, 120))
        for pacote in out.encode(av.VideoFrame.from_image(marcada)):
            saida.mux(pacote)
        linhas.append({"quadro": i, "fissura": r.tem_fissura, "confianca": round(r.confianca, 3), "tempo_ms": round(r.tempo_ms, 1)})
        quadros += 1
    if out is not None:
        for pacote in out.encode():
            saida.mux(pacote)
    saida.close()
    entrada.close()
    media = t_total / max(1, quadros)
    return {"quadros": linhas, "video": saida_buf.getvalue(), "tempo_medio_ms": round(media, 1),
            "fps": round(1000 / media, 1) if media else 0.0,
            "quadros_com_fissura": sum(l["fissura"] for l in linhas)}
