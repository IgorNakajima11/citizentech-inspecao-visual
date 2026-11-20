import json
import random
from pathlib import Path

import libarchive

RAIZ = Path(__file__).resolve().parents[1]
RAR = RAIZ.parent / "dados_brutos" / "concrete.rar"
DESTINO = RAIZ / "dados"
POR_CLASSE = 1500
SEMENTE = 42
TAMANHO_BLOCO = 250


def main() -> None:
    nomes = {"Positive": [], "Negative": []}
    with libarchive.file_reader(str(RAR)) as arq:
        for e in arq:
            partes = Path(e.pathname).parts
            if e.isfile and len(partes) >= 2 and partes[-2] in nomes:
                nomes[partes[-2]].append(e.pathname)
    rng = random.Random(SEMENTE)
    escolhidos = {}
    for classe, lista in nomes.items():
        lista.sort()
        escolhidos.update({p: classe for p in rng.sample(lista, POR_CLASSE)})
    divisao = {}
    for classe in nomes:
        blocos = {}
        for p in sorted(p for p, c in escolhidos.items() if c == classe):
            blocos.setdefault(int(Path(p).stem.split("_")[0]) // TAMANHO_BLOCO, []).append(p)
        ids = sorted(blocos)
        rng.shuffle(ids)
        n, acumulado = POR_CLASSE, 0
        for b in ids:
            destino = "treino" if acumulado < 0.6 * n else ("validacao" if acumulado < 0.8 * n else "teste")
            for p in blocos[b]:
                divisao[p] = destino
            acumulado += len(blocos[b])
    with libarchive.file_reader(str(RAR)) as arq:
        for e in arq:
            if e.pathname in escolhidos:
                classe = "fissura" if escolhidos[e.pathname] == "Positive" else "sem_fissura"
                pasta = DESTINO / divisao[e.pathname] / classe
                pasta.mkdir(parents=True, exist_ok=True)
                with open(pasta / Path(e.pathname).name, "wb") as f:
                    for bloco in e.get_blocks():
                        f.write(bloco)
    resumo = {"total_na_base": {c: len(v) for c, v in nomes.items()}, "amostra_por_classe": POR_CLASSE,
              "semente": SEMENTE, "tamanho_bloco": TAMANHO_BLOCO, "divisao": {d: sum(1 for v in divisao.values() if v == d)
                                             for d in ("treino", "validacao", "teste")}}
    (DESTINO / "resumo.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")
    print(resumo)


if __name__ == "__main__":
    main()
