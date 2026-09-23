import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class TrackerComportamental:
    def __init__(self, model_path=None, model_lixo_path=None, model_geral_path='yolo11s.pt',
                 limite_tempo_estatico_segundos=3, distancia_proximidade_pixels=120):
        """
        Inicializa o rastreador comportamental com suporte dual de modelos (YOLO11 COCO + Modelo Customizado de Lixo).

        :param model_path: Mantido para retrocompatibilidade com notebooks existentes.
        :param model_lixo_path: Caminho para os pesos customizados do Roboflow (best.pt).
        :param model_geral_path: Modelo oficial YOLO11 para detecção de pessoas e animais.
        :param limite_tempo_estatico_segundos: Tempo em segundos que o objeto deve ficar parado após a pessoa se afastar.
        :param distancia_proximidade_pixels: Distância máxima em pixels para considerar interação entre pessoa e objeto.
        """
        path_lixo = model_lixo_path or model_path
        if not path_lixo:
            raise ValueError("É necessário fornecer o caminho do modelo de lixo (model_path ou model_lixo_path).")

        # Seleciona explicitamente GPU CUDA se disponível
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # Inicializa ambos os modelos da Ultralytics
        self.model_lixo = YOLO(path_lixo)
        self.model_geral = YOLO(model_geral_path)

        self.limite_tempo_estatico = limite_tempo_estatico_segundos
        self.distancia_proximidade = distancia_proximidade_pixels
        self.tempo_parado = 0
        self.tempo_estatico_geral = time.time()
        self.tempo_estatico = time.time()
        self.pessoa_detectada = False

        # Dicionário de estado dos objetos/animais rastreados:
        # id_track -> { 'centro': (x,y), 'tempo_inicio_estatico': float, 'interacao_humana': bool, 'alerta': bool, 'tipo': str }
        self.historico_objetos = {}
        # Histórico de rastreamento para controle de tempo: {track_id: timestamp_inicial}
        self.tempo_estatico_objetos = {}

        self.tempo_ini = time.time()
        
    def _calcular_centro(self, box):
        """Calcula o ponto central (x, y) de um Bounding Box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _calcular_distancia(self, p1, p2):
        """Calcula a distância euclidiana entre dois pontos."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def processar_frame(self, frame):
        """
        Processa um frame individual, realiza a detecção combinada e aplica a regra comportamental.
        """
        tempo_atual = time.time()
        frame_desenho = frame.copy()

        # -------------------------------------------------------------
        # 1. Executa Rastreamento Geral (Pessoas e Animais)
        # -------------------------------------------------------------
        resultados_geral = self.model_geral.track(frame, persist=True, verbose=False, conf=0.25, imgsz=1280, device=self.device)

        if resultados_geral and resultados_geral[0].boxes is not None and resultados_geral[0].boxes.id is not None:
          boxes_geral = resultados_geral[0].boxes.xyxy.cpu().numpy()
          clss_geral = resultados_geral[0].boxes.cls.cpu().numpy().astype(int)
          ids_geral = resultados_geral[0].boxes.id.cpu().numpy().astype(int)
          self.tempo_parado = time.time() - self.tempo_ini
          nomes_classes = resultados_geral[0].names

          for box, cls_idx, track_id in zip(boxes_geral, clss_geral, ids_geral):
            nome_classe = nomes_classes.get(cls_idx, "Geral")
            cor = (255, 200, 0)  # Azul Ciano
            label_geral = f"{nome_classe} #{track_id}"
            
            # Se detectar uma pessoa, então define o tempo
            if(nome_classe == 'person'): 
              self.pessoa_detectada = True
              self.tempo_estatico_geral = time.time() - self.tempo_ini
            else:
              self.tempo_estatico_geral = self.tempo_parado

            if(nome_classe != 'person'):
              cor = (0, 165, 255)  # Laranja (Monitorando descarte)
              label_geral = f"{nome_classe} #{track_id} (Aguardando {int(self.tempo_estatico_geral - self.tempo_parado)}s)"

            # Se uma pessoa for detectada e o tempo for maior que 3s, então adiciona o alerta para os animais
            if(self.pessoa_detectada and int(self.tempo_estatico_geral - self.tempo_parado) > self.limite_tempo_estatico):
              cor_box = (0, 0, 255)  # Vermelho Alerta
              label = f"ALERTA: Descarte Irregular - {nome_obj} #{track_id}"
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame_desenho, label_geral, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


        # -------------------------------------------------------------
        # 2. Rastreamento do Modelo de Lixo
        # -------------------------------------------------------------
        resultados_lixo = self.model_lixo.track(frame, persist=True, verbose=False, conf=0.50, imgsz=1280, device=self.device)

        if resultados_lixo and resultados_lixo[0].boxes is not None and resultados_lixo[0].boxes.id is not None:
          boxes_lixo = resultados_lixo[0].boxes.xyxy.cpu().numpy()
          clss_lixo = resultados_lixo[0].boxes.cls.cpu().numpy().astype(int)
          ids_lixo = resultados_lixo[0].boxes.id.cpu().numpy().astype(int)
          nomes_classes_lixo = resultados_lixo[0].names

          for box, cls_idx, track_id in zip(boxes_lixo, clss_lixo, ids_lixo):
            centro_objeto = self._calcular_centro(box)
            nome_obj = nomes_classes_lixo.get(cls_idx, "Lixo")

            # Se detectar uma pessoa, então define o tempo
            if(self.pessoa_detectada): 
              self.tempo_estatico = time.time() - self.tempo_ini
            else:
              self.tempo_estatico = self.tempo_parado

            cor_box = (0, 165, 255)  # Laranja (Monitorando descarte)
            label = f"{nome_obj} #{track_id} (Aguardando {int(self.tempo_estatico - self.tempo_parado)}s)"

            # Se uma pessoa for detectada e o tempo for maior que 3s, então adiciona o alerta para as objetos
            if(self.pessoa_detectada and int(self.tempo_estatico - self.tempo_parado) > self.limite_tempo_estatico):
              cor_box = (0, 0, 255)  # Vermelho Alerta
              label = f"ALERTA: Descarte Irregular - {nome_obj} #{track_id}"
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor_box, 2)
            cv2.putText(frame_desenho, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_box, 2)

        return frame_desenho, self.historico_objetos
