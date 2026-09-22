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

# 1. Monta o Google Drive
drive.mount('/content/drive')

# 2. Atualiza e importa o repositório do projeto
REPO_DIR = '/content/projeto_tcc'
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

os.system(f'git clone https://github.com/marcosptz/tcc-sistemas-informacao.git {REPO_DIR}')

if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

from logic import TrackerComportamental

# 3. Inicializa o Tracker com o modelo treinado no YOLO11
MODELO_PATH = '/content/drive/MyDrive/TCC_Resultados/treino_lixo_yolo11/weights/best.pt'

tracker = TrackerComportamental(
    model_path=MODELO_PATH,
    limite_tempo_estatico_segundos=3
)

# 4. Função principal de processamento (Vídeo Gravado ou Câmera IP Ao Vivo)
def processar_midia(video_path, url_camera_ip, duracao_stream_segundos=15):
    """
    Processa um arquivo de vídeo enviado OU um link RTSP/HTTP de câmera IP.
    """
    # Define a fonte de entrada
    if url_camera_ip and url_camera_ip.strip():
        fonte = url_camera_ip.strip()
        is_stream = True
    elif video_path is not None:
        fonte = video_path
        is_stream = False
    else:
        return None

    cap = cv2.VideoCapture(fonte)

    if not cap.isOpened():
        print(f"Erro ao abrir a fonte de vídeo: {fonte}")
        return None

    # Obtém propriedades do vídeo/stream
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    if fps <= 0 or fps > 60:
        fps = 30

    temp_output = '/content/temp_processado.mp4'
    final_output = '/content/video_processado.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Se for stream ao vivo, limita a quantidade de frames gravados
    max_frames = int(fps * duracao_stream_segundos) if is_stream else float('inf')
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Processa o frame com a IA de detecção e rastreamento
        frame_anotado, _ = tracker.processar_frame(frame)
        out.write(frame_anotado)
        frame_count += 1

    cap.release()
    out.release()

    # Recodifica o vídeo para H.264 para reprodução nativa no navegador (Gradio)
    os.system(f'ffmpeg -y -i {temp_output} -vcodec libx264 {final_output}')

    return final_output

# 5. Interface Gradio
demo = gr.Interface(
    fn=processar_midia,
    inputs=[
        gr.Video(label="Opção 1: Upload de Vídeo MP4 (Arquivo)"),
        gr.Textbox(
            label="Opção 2: Link da Câmera IP / RTSP (Transmissão Ao Vivo)",
            placeholder="Ex: rtsp://admin:senha@192.168.1.100:554/stream1 ou http://192.168.1.50:8080/video"
        ),
        gr.Slider(
            minimum=5,
            maximum=60,
            value=15,
            step=5,
            label="Duração da captura do Stream IP (segundos)"
        )
    ],
    outputs=gr.Video(label="Resultado Processado com Detecção do TCC"),
    title="Sistema de Monitoramento CFTV com IA - Detecção de Descarte Irregular",
    description="Escolha entre fazer o upload de um vídeo gravado ou colar a URL RTSP/HTTP de uma Câmera IP ao vivo."
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
