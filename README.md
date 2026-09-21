# 🎥 Sistema de Monitoramento com IA - Detecção e Alerta de Descarte Indevido de Objetos

Este projeto foi desenvolvido como trabalho de conclusão de curso (TCC) e consiste em um **sistema de monitoramento inteligente para câmeras CFTV**, focado na detecção de pessoas, rastreamento comportamental e identificação de objetos descartados/estáticos (lixo, sacolas, recipientes e garrafas).

---

## 🚀 Executar Diretamente no Google Colab

Clique no botão abaixo para abrir o notebook pronto para execução no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcosptz/tcc-sistemas-informacao/blob/main/TCC.ipynb)

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **YOLOv8 / YOLO11 / YOLO26 (Ultralytics):** Visão computacional para detecção e rastreamento (*Tracking*) de objetos e pessoas.
* **OpenCV:** Processamento e manipulação dos quadros dos vídeos.
* **Gradio:** Interface web interativa no navegador para envio de vídeos e visualização em tempo real.
* **FFmpeg:** Codificação de saída de vídeo compatível com navegadores web (H.264).

---

## 📋 Funcionalidades

1. **Rastreamento de Pessoas e Objetos:** Identifica a presença de transeuntes e itens deixados no perímetro.
2. **Lógica Temporizada para Objetos Estáticos:** Alerta e contabiliza o tempo que um objeto permanece imóvel na cena após ser solto.
3. **Interface Simples via Web:** Envio de arquivo MP4 e geração automática do vídeo processado com as bounding boxes e alertas visualmente anotados.

---

## ⚙️ Como Executar Passo a Passo

### Opção 1: Via Google Colab (Recomendado)

1. Clique no botão **"Open in Colab"** no topo deste README.
2. Vá no menu do Colab em **Ambiente de execução > Alterar o tipo de ambiente de execução** e escolha a opção de **GPU (T4)**.
3. Execute a célula principal do Notebook.
4. Ao final da execução, será gerado um **Link Público do Gradio** (ex: `https://xxxx.gradio.live`).
5. Acesse o link, faça o upload do vídeo de teste (ex: `1000129892.mp4`) e clique em **Submit**.

---

### Opção 2: Código da Célula Única no Colab

Se preferir rodar manualmente em qualquer bloco de notas do Colab, utilize o código abaixo:

```python
!pip install -q --upgrade ultralytics gradio opencv-python

import os
import shutil
import sys
import cv2
import gradio as gr
from google.colab import drive
from ultralytics import YOLO

drive.mount('/content/drive')

# 1. Atualiza o repositório GitHub
if os.path.exists('/content/projeto_tcc'):
  shutil.rmtree('/content/projeto_tcc')

!git clone https://github.com/marcosptz/tcc-sistemas-informacao.git /content/projeto_tcc

if '/content/projeto_tcc' not in sys.path:
  sys.path.append('/content/projeto_tcc')

from logic import TrackerComportamental

# -------------------------------------------------------------
# Escolha qual modelo quer usar (Descomente apenas uma das opções):
# -------------------------------------------------------------

# Opção A: YOLOv8s, YOLO11s, YOLO26s (Nome correto: sem o 'v')
# MODEL_PATH = 'yolo26s.pt'

# Opção B: Seu modelo customizado de Lixo treinado no Roboflow
MODEL_PATH = '/content/drive/MyDrive/TCC_Resultados/treino_lixo_yolo11/weights/best.pt'

# 2. Inicializa o Tracker passando a STRING com o caminho do modelo
tracker = TrackerComportamental(
    model_path=MODEL_PATH, limite_tempo_estatico_segundos=3
)

# 3. DEFINE A FUNÇÃO PROCESSAR_VIDEO_COLAB
def processar_video_colab(video_path):
  if video_path is None:
    return None

  cap = cv2.VideoCapture(video_path)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

  temp_output = '/content/temp_processado.mp4'
  final_output = '/content/video_processado.mp4'

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Processa o frame com a lógica de detecção e tempo estático
    frame_anotado, _ = tracker.processar_frame(frame)
    out.write(frame_anotado)

  cap.release()
  out.release()

  # Converte para H.264 usando FFmpeg para compatibilidade com o navegador no Gradio
  os.system(f'ffmpeg -y -i {temp_output} -vcodec libx264 {final_output}')

  return final_output


# 4. Inicializa e lança a Interface do Gradio
demo = gr.Interface(
    fn=processar_video_colab,
    inputs=gr.Video(label="Upload do Vídeo de Teste"),
    outputs=gr.Video(label="Vídeo com Detecção e Alertas"),
    title="Sistema de Monitoramento com IA - COCO Model",
)

demo.launch(share=True, debug=True)
