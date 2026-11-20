import json
import pickle
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fastembed import ImageEmbedding  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
RES = RAIZ / "resultados"
MODELOS = {"resnet50": "Qdrant/resnet50-onnx", "clip_vit_b32": "Qdrant/clip-ViT-B-32-vision"}
CLASSES = ["sem_fissura", "fissura"]


def listar(conjunto: str) -> tuple[list[Path], np.ndarray]:
    caminhos, rotulos = [], []
    for y, c in enumerate(CLASSES):
        for p in sorted((DADOS / conjunto / c).glob("*.jpg")):
            caminhos.append(p)
            rotulos.append(y)
    return caminhos, np.array(rotulos)


def extrair(modelo: ImageEmbedding, caminhos: list[Path]) -> np.ndarray:
    return np.array(list(modelo.embed([str(p) for p in caminhos], batch_size=32)), dtype=np.float32)


def escolher_limiar(prob: np.ndarray, y: np.ndarray) -> float:
    candidatos = np.arange(0.2, 0.81, 0.05)
    f1s = [f1_score(y, (prob >= t).astype(int)) for t in candidatos]
    return float(round(candidatos[int(np.argmax(f1s))], 2))


def medir_latencia(modelo: ImageEmbedding, clf, caminhos: list[Path], n: int = 50) -> dict:
    imgs = [Image.open(p).convert("RGB") for p in caminhos[:n]]
    list(modelo.embed([imgs[0]]))
    tempos = []
    for im in imgs:
        t0 = time.perf_counter()
        v = np.array(list(modelo.embed([im])), dtype=np.float32)
        clf.predict_proba(v)
        tempos.append(time.perf_counter() - t0)
    media = float(np.mean(tempos))
    return {"latencia_media_ms": round(1000 * media, 1), "latencia_p95_ms": round(1000 * float(np.percentile(tempos, 95)), 1),
            "fps": round(1 / media, 1), "imagens_medidas": n}


def main() -> None:
    tr, ytr = listar("treino")
    va, yva = listar("validacao")
    te, yte = listar("teste")
    resultados = {"equipamento": {"cpu": next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
                                  "gpu": "nenhuma (só CPU)", "sistema": platform.platform()},
                  "resolucao_entrada": "227x227 da base, redimensionada pelo pré-processamento de cada modelo para 224x224",
                  "n": {"treino": len(tr), "validacao": len(va), "teste": len(te)}, "modelos": {}}
    for nome, mid in MODELOS.items():
        print("modelo", nome, flush=True)
        m = ImageEmbedding(mid)
        t0 = time.perf_counter()
        xtr, xva, xte = extrair(m, tr), extrair(m, va), extrair(m, te)
        t_extr = time.perf_counter() - t0
        melhor = None
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, max_iter=3000).fit(xtr, ytr)
            f1 = f1_score(yva, clf.predict(xva))
            if melhor is None or f1 > melhor[0]:
                melhor = (f1, c, clf)
        _, c, clf = melhor
        limiar = escolher_limiar(clf.predict_proba(xva)[:, 1], yva)
        prob = clf.predict_proba(xte)[:, 1]
        pred = (prob >= limiar).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average=None, labels=[0, 1])
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        erros = [{"arquivo": str(te[i].relative_to(RAIZ)), "real": CLASSES[yte[i]], "previsto": CLASSES[pred[i]],
                  "confianca_fissura": round(float(prob[i]), 3)} for i in np.where(pred != yte)[0]]
        lat = medir_latencia(m, clf, te)
        resultados["modelos"][nome] = {
            "id": mid, "C": c, "limiar": limiar, "tempo_extracao_s": round(t_extr, 1),
            "fissura": {"precisao": round(float(p[1]), 4), "recall": round(float(r[1]), 4), "f1": round(float(f[1]), 4)},
            "sem_fissura": {"precisao": round(float(p[0]), 4), "recall": round(float(r[0]), 4), "f1": round(float(f[0]), 4)},
            "acuracia": round(float((pred == yte).mean()), 4), "matriz_confusao": cm.tolist(), "erros": erros, **lat}
        with open(RAIZ / "modelo" / f"classificador_{nome}.pkl", "wb") as fh:
            pickle.dump({"modelo_embedding": mid, "classificador": clf, "limiar": limiar, "classes": CLASSES}, fh)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["sem fissura", "fissura"])
        ax.set_yticks([0, 1], ["sem fissura", "fissura"])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title(f"{nome} (teste, limiar {limiar})", fontsize=9)
        fig.tight_layout()
        fig.savefig(RES / f"matriz_confusao_{nome}.png", dpi=150)
        plt.close(fig)
        print(nome, json.dumps({k: v for k, v in resultados["modelos"][nome].items() if k != "erros"}), flush=True)
    (RES / "metricas.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
