# CitizenTech: inspeção de fissuras por imagem

Global Solution 2026.2, disciplina Visão Computacional.
Integrante: **Igor Thiago Nakajima Vieira**, RM 563632 (trabalho individual).

Módulo de visão computacional do CitizenTech para **triagem de locais para inspeção**:
- **Foto:** o cidadão ou o fiscal envia a foto de uma superfície de concreto, e o sistema marca as regiões com
  possível fissura, com a confiança do modelo.
- **Vídeo:** o módulo também processa vídeo quadro a quadro, para simular o monitoramento urbano.

A presença de fissura na imagem, sozinha, **não indica gravidade nem risco estrutural**. O resultado só ajuda a
priorizar a visita de um técnico.

Relatório com métricas, latência, justificativas e análise de erros:
[relatorio/Relatorio_VC_Inspecao_Fissuras_Igor_Nakajima_563632.pdf](relatorio/Relatorio_VC_Inspecao_Fissuras_Igor_Nakajima_563632.pdf)

## Integração com o CitizenTech

O CitizenTech ([citizentech-triagem-web](https://github.com/IgorNakajima11/citizentech-triagem-web)) recebe relatos
de problemas urbanos. Este módulo acrescenta a análise da foto do relato:
- quando a foto mostra fissura, o local é sugerido para inspeção;
- quando não mostra, o relato segue o fluxo normal.

O módulo é independente: a função `inspecao.modelo.analisar(imagem)` recebe uma imagem PIL e devolve:
- se há fissura;
- a confiança;
- as regiões marcadas;
- o tempo gasto.

## Estrutura

```
app.py                         tela Streamlit: analisar imagem e monitoramento por vídeo
inspecao/modelo.py             inferência: imagem inteira ou em blocos, vídeo quadro a quadro, marcação visual
scripts/preparar_dados.py      amostra da base, divisão treino/validação/teste por blocos de numeração
scripts/treinar_avaliar.py     extração de características, treino, limiar, métricas, matriz de confusão, latência
scripts/gerar_demonstracoes.py paredes simuladas (mosaicos) e vídeo de monitoramento com imagens de teste
modelo/                        classificadores treinados (ResNet50 e CLIP, para comparação)
resultados/                    metricas.json, matrizes de confusão e exemplos de erros
demonstracao/                  imagens e vídeo de exemplo
relatorio/                     relatório em PDF
```

## Dados

**Base:** Concrete Crack Images for Classification (Özgenel, 2019), em
https://data.mendeley.com/datasets/5y9wdsg2zt/1, licença CC BY 4.0.
- 40.000 imagens de 227x227 pixels, sendo 20.000 com fissura e 20.000 sem.
- Usamos uma amostra de 1.500 por classe, com semente fixa, como o enunciado permite para caber no prazo.

**Divisão:** treino 60%, validação 20%, teste 20%.
- A base não informa de qual foto original veio cada recorte.
- Para reduzir o risco de recortes da mesma foto caírem em conjuntos diferentes, a divisão é feita por **blocos de
  250 números consecutivos**: imagens vizinhas na numeração vão sempre para o mesmo conjunto.
- Isso reduz o vazamento, mas não o elimina. A limitação está discutida no relatório.

**Pré-processamento:** cada modelo redimensiona para 224x224 e normaliza com a média e o desvio do ImageNet.
- Em fotos maiores, a imagem é dividida em blocos de cerca de 227x227, e cada bloco é classificado.
- Isso permite localizar a região da fissura.

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# só se quiser refazer o treino (a base de 241 MB precisa estar em ../dados_brutos/concrete.rar):
python scripts/preparar_dados.py
python scripts/treinar_avaliar.py
python scripts/gerar_demonstracoes.py

streamlit run app.py     # abre em http://localhost:8501
```

Os classificadores treinados já estão em `modelo/`, então a tela funciona sem refazer o treino. Na primeira execução,
o ResNet50 em ONNX (cerca de 100 MB) é baixado automaticamente.

## Resultados (conjunto de teste, 592 imagens)

| Modelo | Precisão (fissura) | Recall (fissura) | F1 (fissura) | Acurácia | Latência por imagem | FPS |
|---|---|---|---|---|---|---|
| **ResNet50 + regressão logística (final)** | 99,0% | 99,3% | 99,2% | 99,2% | 29.3 ms | 34.1 |
| CLIP ViT-B/32 + regressão logística | 99,3% | 99,0% | 99,2% | 99,2% | 43.4 ms | 23.1 |

- **Matriz de confusão do ResNet50:** [[290, 3], [2, 297]] (linhas = real [sem fissura, fissura]; colunas = previsto).
  As figuras estão em `resultados/`.
- **Limiar:** 0.45, escolhido na validação pelo melhor F1. É ajustável na tela.
- **Equipamento:** 13th Gen Intel(R) Core(TM) i5-1334U, sem GPU.
- **Vídeo:** 40 quadros de 454x454, com 4 blocos por quadro. Levou 92.1 ms por quadro
  (10.9 FPS), e 15 quadros foram marcados com possível fissura.
- **Por que o ResNet50:** a qualidade foi igual à do CLIP, e ele é cerca de 1,5 vez mais rápido.

**Erros (5 em 592):**
- 3 falsos alarmes:
  - manchas e textura;
  - poros escuros;
  - um objeto linear escuro parecido com fio, que o CLIP também errou.
- 2 fissuras perdidas, em superfície lisa e escura, com confiança de 0,41 e 0,42, logo abaixo do limiar.

A grade com os erros está em `resultados/erros_resnet50.png`, e a análise completa está no relatório.

**Limitações:** a base tem recortes próximos e bem enquadrados, então não se deve esperar 99% em fotos de rua. Também
pode haver vazamento residual entre conjuntos, a localização é por blocos (não por caixas) e o modelo não mede
gravidade.

