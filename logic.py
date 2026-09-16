import cv2
import time
import numpy as np
from ultralytics import YOLO

class TrackerComportamental:
    def __init__(self, model_path, limite_tempo_estatico_segundos=5):
        self.model = YOLO(model_path)
        self.limite_tempo = limite_tempo_estatico_segundos
        self.objetos_rastreados = {}  # {id_objeto: timestamp_inicio}

    def processar_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        alertas = []
        
        if results[0].boxes is None or results[0].boxes.id is None:
            return frame, alertas

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        cls_ids = results[0].boxes.cls.int().cpu().numpy()
        names = self.model.names

        tempo_atual = time.time()

        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            classe = names[cls_id]
            x1, y1, x2, y2 = map(int, box)

            # Foco na detecção de descarte (lixo) ou animais abandonados
            if classe in ['sacola_lixo', 'entulho_lixo', 'cao', 'gato']:
                if track_id not in self.objetos_rastreados:
                    self.objetos_rastreados[track_id] = tempo_atual

                tempo_permanencia = tempo_atual - self.objetos_rastreados[track_id]

                # Alerta se o objeto/animal permanecer parado além do tempo limite
                if tempo_permanencia >= self.limite_tempo:
                    alerta_msg = f"ALERTA: Possovel {classe} estatico/abandonado (ID: {track_id})"
                    alertas.append(alerta_msg)
                    
                    # Destaque em vermelho no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, alerta_msg, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Objeto recente (verde)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Veículos e pessoas (azul)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame, alertas