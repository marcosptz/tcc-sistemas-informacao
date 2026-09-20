# 🎥 Sistema de Monitoramento com IA - Detecção e Alerta de Descarte Indevido de Objetos

Este projeto foi desenvolvido como trabalho de conclusão de curso (TCC) e consiste em um **sistema de monitoramento inteligente para câmeras CFTV**, focado na detecção de pessoas, rastreamento comportamental e identificação de objetos descartados/estáticos (lixo, sacolas, recipientes e garrafas).

---

## 🚀 Executar Diretamente no Google Colab

Clique no botão abaixo para abrir o notebook pronto para execução no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcosptz/tcc-sistemas-informacao/blob/main/TCC.ipynb)

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **YOLOv8 / YOLO11 (Ultralytics):** Visão computacional para detecção e rastreamento (*Tracking*) de objetos e pessoas.
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
# 1. Instalação e Atualização de Dependências
!pip install -q --upgrade ultralytics gradio opencv-python

import os
import shutil
import sys
import cv2
import gradio as gr
from google.colab import drive

# 2. Atualização Automática do Repositório GitHub
REPO_DIR = '/content/projeto_tcc'
if os.path.exists(REPO_DIR):
  shutil.rmtree(REPO_DIR)

!git clone [https://github.com/marcosptz/tcc-sistemas-informacao.git](https://github.com/marcosptz/tcc-sistemas-informacao.git) {REPO_DIR}

if REPO_DIR not in sys.path:
  sys.path.append(REPO_DIR)

from logic import TrackerComportamental
from ultralytics import YOLO

# 3. Inicialização do Modelo YOLO (Defina 'yolo11s.pt' ou o caminho do seu 'best.pt')

# -------------------------------------------------------------
# Escolha qual modelo quer usar (Descomente apenas uma das opções):
# -------------------------------------------------------------

# Opção A: yolov8s, yolo11s, yolo26s (Nome correto: sem o 'v')
MODEL_PATH = 'yolo26s.pt'

# Opção B: Seu modelo customizado de Lixo treinado no Roboflow
# MODEL_PATH = '/content/drive/MyDrive/TCC_Resultados/treino_lixo_v1/weights/best.pt'

tracker = TrackerComportamental(
    model_path=MODELO_PATH, limite_tempo_estatico_segundos=3
)
model = YOLO(MODELO_PATH)


# 4. Função para Processamento do Vídeo
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

    # Inferência com resolução para CFTV (imgsz=1024)
    results = model.track(frame, persist=True, conf=0.10, imgsz=1024)
    frame_anotado = results[0].plot()
    out.write(frame_anotado)

  cap.release()
  out.release()

  # Converte o vídeo gerado para codificação H.264 para reprodução no navegador
  os.system(f'ffmpeg -y -i {temp_output} -vcodec libx264 {final_output}')
  return final_output


# 5. Lançamento da Interface Web Gradio
demo = gr.Interface(
    fn=processar_video_colab,
    inputs=gr.Video(label="Upload do Vídeo de Teste"),
    outputs=gr.Video(label="Vídeo com Detecção e Alertas"),
    title="Sistema de Monitoramento com IA - TCC",
)

demo.launch(share=True, debug=True)
