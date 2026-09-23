import math
import time
import cv2
import numpy as np
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

        # Inicializa ambos os modelos da Ultralytics
        self.model_lixo = YOLO(path_lixo)
        self.model_geral = YOLO(model_geral_path)

        self.limite_tempo_estatico = limite_tempo_estatico_segundos
        self.distancia_proximidade = distancia_proximidade_pixels

        # Dicionário de estado dos objetos/animais rastreados:
        # id_track -> { 'centro': (x,y), 'tempo_inicio_estatico': float, 'interacao_humana': bool, 'alerta': bool, 'tipo': str }
        self.historico_objetos = {}
        # Histórico de rastreamento para controle de tempo: {track_id: timestamp_inicial}
        self.tempo_estatico_objetos = {}

        self.tempo_ini = time.time()

        # Mapeamento de classes do modelo COCO para Pessoas e Animais
        self.CLASSES_GERAIS_INTERESSE = {
            16: 'Cão',
            17: 'Gato',
            18: 'Pessoa'
        }

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
        existe_pessoa = 0

        # -------------------------------------------------------------
        # 1. Executa Rastreamento Geral (Pessoas e Animais)
        # -------------------------------------------------------------
        resultados_geral = self.model_geral.track(frame, persist=True, verbose=False, conf=0.25)

        centros_pessoas = []
        centros_animais = []
        centros_objetos = []

        if resultados_geral and resultados_geral[0].boxes is not None and resultados_geral[0].boxes.id is not None:
          boxes_geral = resultados_geral[0].boxes.xyxy.cpu().numpy()
          clss_geral = resultados_geral[0].boxes.cls.cpu().numpy().astype(int)
          ids_geral = resultados_geral[0].boxes.id.cpu().numpy().astype(int)
          # self.tempo_estatico_objetos[track_id] = tempo_atual
          # self.tempo_estatico = tempo_atual
          self.tempo_parado = time.time() - self.tempo_ini
          nome_classe = 'Geral'
          cor = (255, 200, 0)  # Ciano/Amarelo

          for box, cls_idx, track_id in zip(boxes_geral, clss_geral, ids_geral):
            # Desenha Bounding Box de pessoas/animais
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame_desenho, f"{nome_classe} #{track_id}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


        resultados_lixo = self.model_lixo.track(frame, persist=True, verbose=False, conf=0.20, imgsz=1280)

        if resultados_lixo and resultados_lixo[0].boxes is not None and resultados_lixo[0].boxes.id is not None:
          boxes_lixo = resultados_lixo[0].boxes.xyxy.cpu().numpy()
          clss_lixo = resultados_lixo[0].boxes.cls.cpu().numpy().astype(int)
          ids_lixo = resultados_lixo[0].boxes.id.cpu().numpy().astype(int)
          nomes_classes_lixo = resultados_lixo[0].names
          # self.tempo_estatico = tempo_atual

          for box, cls_idx, track_id in zip(boxes_lixo, clss_lixo, ids_lixo):
            centro_objeto = self._calcular_centro(box)
            nome_obj = nomes_classes_lixo.get(cls_idx, "Lixo")
            centros_objetos.append({'id': track_id, 'centro': centro_objeto, 'box': box, 'tipo': nome_obj})
            self.tempo_estatico = time.time() - self.tempo_ini
            existe_pessoa = existe_pessoa + 1
            # cor_box = (0, 0, 255)  # Vermelho Alerta
            # label = f"ALERTA: Descarte Irregular #{track_id}"
            cor_box = (0, 165, 255)  # Laranja (Monitorando descarte)
            label = f"{nome_obj} #{track_id} (Aguardando {int(self.tempo_estatico - self.tempo_parado)}s)"
            # print(f"Tempo atual {tempo_parado - tempo_atual}")
            if(int(self.tempo_estatico - self.tempo_parado) > self.limite_tempo_estatico):
              cor_box = (0, 0, 255)  # Vermelho Alerta
              label = f"ALERTA: Descarte Irregular #{track_id}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor_box, 2)
            cv2.putText(frame_desenho, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_box, 2)

        return frame_desenho, self.historico_objetos
