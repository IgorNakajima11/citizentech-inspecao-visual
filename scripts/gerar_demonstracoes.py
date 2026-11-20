import random
from pathlib import Path

import av
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
TESTE = RAIZ / "dados" / "teste"
DEMO = RAIZ / "demonstracao"
DEMO.mkdir(exist_ok=True)


def main() -> None:
    rng = random.Random(7)
    fis = sorted((TESTE / "fissura").glob("*.jpg"))
    sem = sorted((TESTE / "sem_fissura").glob("*.jpg"))
    for k in range(3):
        mosaico = Image.new("RGB", (227 * 3, 227 * 2))
        escolhas = rng.sample(fis, 2) + rng.sample(sem, 4)
        rng.shuffle(escolhas)
        for n, p in enumerate(escolhas):
            mosaico.paste(Image.open(p).convert("RGB"), ((n % 3) * 227, (n // 3) * 227))
        mosaico.save(DEMO / f"parede_simulada_{k + 1}.jpg", quality=92)
    sequencia = rng.sample(fis, 15) + rng.sample(sem, 25)
    rng.shuffle(sequencia)
    saida = av.open(str(DEMO / "monitoramento_simulado.mp4"), mode="w")
    s = saida.add_stream("libx264", rate=4)
    s.width, s.height, s.pix_fmt = 454, 454, "yuv420p"
    for p in sequencia:
        img = Image.open(p).convert("RGB").resize((454, 454))
        for pacote in s.encode(av.VideoFrame.from_image(img)):
            saida.mux(pacote)
    for pacote in s.encode():
        saida.mux(pacote)
    saida.close()
    print("ok")


if __name__ == "__main__":
    main()
