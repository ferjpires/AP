# Ficha CNN RGB Multiclass 2026 — Passo a passo

## Como usar

1. Abre o notebook `Ficha_CNN_RGB_multiclass_2026_resolucao.ipynb`.
2. De preferência, usa Google Colab com GPU:
   - Runtime > Change runtime type > GPU.
3. Se tiveres o ficheiro `notebooksDataset(1).zip`, faz upload para a mesma pasta do notebook.
   - O notebook extrai automaticamente o `cifar.tgz` e os modelos `.pth`.
   - Se não tiveres o zip, o notebook tenta descarregar o `cifar.tgz`.
4. Executa as células por ordem.
5. Para cumprir a ficha, mantém `RUN_TRAINING = True`.
6. No fim, executa a célula de comparação final e a célula que cria o zip de entrega.

## O que o notebook faz

- T1: prepara o CIFAR-10;
- T2: instala dependências;
- T3: usa batch size 128;
- T4: cria classes, normaliza imagens, prepara CHW e data loaders com holdout;
- T5: mostra métricas e batch de imagens;
- T6: verifica balanceamento;
- T7: define ResNet, CNN1, CNN2, CNN3 e CNN4;
- T8: treina todos os modelos com os hiperparâmetros pedidos;
- T9: avalia modelos, classificação e matriz de confusão;
- T10: faz previsão de uma imagem;
- T11: gera resultados, gráficos e zip de entrega.

## Nota

Se estiveres em CPU, o treino completo pode ser bastante lento. Para testar rapidamente, podes mudar:
`RUN_TRAINING = False`

Nesse modo, o notebook tenta usar os ficheiros `.pth` já fornecidos no zip.