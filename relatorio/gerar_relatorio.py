import json
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
SAIDA_HTML = RAIZ / "relatorio" / "relatorio.html"
SAIDA_PDF = RAIZ / "relatorio" / "Relatorio_VC_Inspecao_Fissuras_Igor_Nakajima_563632.pdf"


def pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def tabela(cab, linhas):
    h = "".join(f"<th>{c}</th>" for c in cab)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>" for l in linhas)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main():
    m = json.loads((RES / "metricas.json").read_text(encoding="utf-8"))
    v = json.loads((RES / "video_desempenho.json").read_text(encoding="utf-8"))
    r, c = m["modelos"]["resnet50"], m["modelos"]["clip_vit_b32"]
    comp = tabela(["Modelo", "Precisão (fissura)", "Recall (fissura)", "F1 (fissura)", "Acurácia", "Latência média",
                   "FPS (1 imagem 227x227)"],
                  [[f"<b>ResNet50 (final)</b>", pct(r["fissura"]["precisao"]), pct(r["fissura"]["recall"]), pct(r["fissura"]["f1"]),
                    pct(r["acuracia"]), f"{r['latencia_media_ms']} ms", r["fps"]],
                   ["CLIP ViT-B/32", pct(c["fissura"]["precisao"]), pct(c["fissura"]["recall"]), pct(c["fissura"]["f1"]),
                    pct(c["acuracia"]), f"{c['latencia_media_ms']} ms", c["fps"]]])
    classes = tabela(["Classe (ResNet50, teste)", "Precisão", "Recall", "F1"],
                     [["Fissura", pct(r["fissura"]["precisao"]), pct(r["fissura"]["recall"]), pct(r["fissura"]["f1"])],
                      ["Sem fissura", pct(r["sem_fissura"]["precisao"]), pct(r["sem_fissura"]["recall"]), pct(r["sem_fissura"]["f1"])]])
    uri = lambda n: (RES / n).as_uri()  # noqa: E731
    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório VC</title><style>
@page {{ size: A4; margin: 2.5cm 2cm 2cm 2.5cm; }}
body {{ font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; }}
h1 {{ font-size: 14pt; margin: 0 0 4pt; }} h2 {{ font-size: 11.5pt; margin: 14pt 0 4pt; text-transform: uppercase; }}
p {{ text-align: justify; margin: 0 0 7pt; }} .sub {{ color: #333; margin-bottom: 12pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4pt 0 8pt; }}
th, td {{ border: 1px solid #555; padding: 3pt 5pt; text-align: left; }} th {{ background: #eee; }}
img {{ max-width: 100%; }} .fig {{ font-size: 8.5pt; color: #333; margin-bottom: 8pt; }} ul {{ margin: 0 0 7pt; }}
.lado {{ display: flex; gap: 8pt; }} .lado img {{ width: 48%; }}
</style></head><body>
<h1>CitizenTech: inspeção de fissuras por imagem</h1>
<div class="sub">Global Solution 2026.2, Visão Computacional. Igor Thiago Nakajima Vieira, RM 563632 (trabalho individual).
Código: https://github.com/IgorNakajima11/citizentech-inspecao-visual</div>

<h2>1 Problema e integração</h2>
<p>O módulo classifica fotos de superfícies de concreto em "com fissura" e "sem fissura" e, em fotos maiores, marca os
blocos da imagem onde há possível fissura, com a confiança do modelo. No CitizenTech, ele analisa a foto enviada junto do
relato do cidadão ou da vistoria e sugere o local para inspeção. Também processa vídeo quadro a quadro, simulando o
monitoramento urbano. A presença de fissura não indica gravidade nem risco estrutural; o resultado só prioriza a visita de
um técnico, e a tela diz isso explicitamente.</p>

<h2>2 Dados e pré-processamento</h2>
<p><b>Base:</b> Concrete Crack Images for Classification (Özgenel, 2019; Mendeley Data; CC BY 4.0), com 40.000 imagens de
227x227 (20.000 por classe). Amostra de 1.500 imagens por classe, semente fixa. <b>Divisão:</b> {m['n']['treino']} de treino,
{m['n']['validacao']} de validação e {m['n']['teste']} de teste. A base não informa de qual foto original veio cada recorte;
para reduzir o risco de recortes da mesma foto em conjuntos diferentes, a divisão é feita por blocos de 250 números
consecutivos (recortes vizinhos na numeração ficam juntos). Isso reduz, mas não garante, a ausência de vazamento.
<b>Pré-processamento:</b> redimensionamento para 224x224 e normalização com média e desvio do ImageNet (feitos pelo próprio
modelo). Fotos maiores são divididas em blocos de cerca de 227x227, o tamanho das imagens de treino.</p>

<h2>3 Arquitetura e justificativa</h2>
<p><b>Transferência de aprendizado:</b> ResNet50 pré-treinada no ImageNet, congelada, como extratora de características
(vetor de 2.048 dimensões, modelo ONNX via fastembed), e uma regressão logística treinada em cima (C escolhido na validação
entre 0,01 e 10; escolhido C = {r['C']}). Treina em cerca de {round(r['tempo_extracao_s'])} s de extração em CPU, sem GPU, e
cabe no prazo. Comparamos com o CLIP ViT-B/32 (transformer de visão) na mesma configuração.</p>
{comp}
<p>A qualidade ficou igual (5 erros em {m['n']['teste']} imagens cada), e o ResNet50 é cerca de 1,5 vez mais rápido. Por
isso ele é o modelo final: em monitoramento contínuo, velocidade é o que diferencia. <b>Limiar de confiança:</b> {r['limiar']},
escolhido na validação pelo melhor F1 entre 0,20 e 0,80. Na tela ele é ajustável: um limiar menor aumenta o recall (menos
fissuras perdidas, mais alarmes falsos), o que faz sentido quando deixar passar uma fissura custa mais que uma visita a mais.</p>

<h2>4 Métricas (conjunto de teste)</h2>
{classes}
<div class="lado"><img src="{uri('matriz_confusao_resnet50.png')}"><img src="{uri('matriz_confusao_clip_vit_b32.png')}"></div>
<div class="fig">Figura 1: matrizes de confusão no teste (esquerda ResNet50, direita CLIP).</div>

<h2>5 Desempenho</h2>
{tabela(["Medida", "Valor"], [
 ["Equipamento", f"{m['equipamento']['cpu']}, {m['equipamento']['gpu']}, Linux"],
 ["Uma imagem 227x227 (pré-processamento + modelo + classificador)", f"{r['latencia_media_ms']} ms em média (p95 {r['latencia_p95_ms']} ms), {r['fps']} FPS"],
 ["Vídeo de monitoramento (40 quadros 454x454, 4 blocos por quadro)", f"{v['tempo_medio_ms']} ms por quadro, {v['fps']} FPS"],
 ["Quadros marcados no vídeo", f"{v['quadros_com_fissura']} de {v['quadros']} (a sequência tem 15 imagens com fissura)"],
])}
<p>A latência cresce com o número de blocos: um quadro de 454x454 tem 4 blocos, por isso cai de {r['fps']} para {v['fps']} FPS.
Para vídeo de câmera em tempo real, a saída seria analisar 1 a cada N quadros (opção já prevista no código) ou usar GPU.</p>

<h2>6 Exemplos de acertos e análise dos erros</h2>
<img src="{uri('exemplo_parede_3_marcada.jpg')}" style="width:55%">
<div class="fig">Figura 2: "parede" simulada com 6 imagens de teste; os 2 blocos com fissura foram marcados (79% e 81%) e os 4
sem fissura, não.</div>
<img src="{uri('erros_resnet50.png')}">
<div class="fig">Figura 3: os 5 erros do ResNet50 no teste (classe real e confiança de fissura).</div>
<ul>
<li><b>Falsos alarmes por textura e manchas (09117):</b> variação de tom e manchas escuras irregulares na superfície.</li>
<li><b>Falsos alarmes por poros (12428):</b> buracos escuros pequenos na superfície, sem fissura.</li>
<li><b>Obstáculo (13694):</b> um objeto linear escuro, parecido com fio ou galho, atravessa a imagem; o modelo, que aprendeu
"linha escura" como fissura, marcou. O CLIP errou a mesma imagem.</li>
<li><b>Fissuras perdidas por iluminação e fundo (00019 e 05437):</b> fissuras nítidas, mas em superfície lisa e mais escura;
a confiança ficou em 0,41 e 0,42, logo abaixo do limiar de 0,45. Um limiar de 0,40 as pegaria, ao custo de mais alarmes.</li>
</ul>
<p>Os erros mostram que o modelo associa "traço escuro com contraste" a fissura. Em foto real de rua, com sombras de
postes, fios, juntas de dilatação e pichação, os falsos alarmes devem aumentar.</p>

<h2>7 Limitações</h2>
<ul>
<li>Todas as imagens são recortes próximos e bem enquadrados de uma única base; o acerto de 99% não deve ser esperado em
fotos de celular tiradas de longe, com perspectiva, sombras e objetos.</li>
<li>Possível vazamento residual entre conjuntos (origem dos recortes desconhecida), o que pode inflar as métricas.</li>
<li>A localização é por blocos (grade), não por caixas delimitadoras precisas; para isso seria preciso um detector treinado
com anotações, como no RDD2022.</li>
<li>O modelo não mede largura, profundidade ou gravidade da fissura.</li>
<li>O vídeo de demonstração é uma sequência de imagens de teste, não uma filmagem real.</li>
</ul>
<p style="font-size:8.5pt">Referência: ÖZGENEL, Ç. F. Concrete Crack Images for Classification. Mendeley Data, v2, 2019.
DOI: 10.17632/5y9wdsg2zt.2. HE, K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.</p>
</body></html>"""
    SAIDA_HTML.write_text(html, encoding="utf-8")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={SAIDA_PDF}", SAIDA_HTML.as_uri()], check=True, capture_output=True)
    print(SAIDA_PDF)


if __name__ == "__main__":
    main()
