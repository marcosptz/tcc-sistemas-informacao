import math
import time
import cv2


class TrackerComportamental:

  def __init__(self, model_path, limite_tempo_estatico_segundos=3):
    from ultralytics import YOLO

    # Desativado o best.pt para carregar o modelo oficial do YOLO
    #self.model = YOLO(model_path)
    # Carregando o modelo oficial pré treinado na COCO
    self.model = YOLO('yolov8s.pt')
    self.limite_segundos = limite_tempo_estatico_segundos

    # Histórico de rastreamento para controle de tempo: {track_id: timestamp_inicial}
    self.tempo_estatico_objetos = {}

  def calcular_distancia(self, p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

  def processar_frame(self, frame):
    # Executa a detecção e o rastreamento do YOLOv8
    results = self.model.track(frame, persist=True, conf=0.25)

    pessoas = []
    animais = []
    objetos = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
      boxes = results[0].boxes.xyxy.cpu().numpy()
      ids = results[0].boxes.id.cpu().numpy()
      clss = results[0].boxes.cls.cpu().numpy()

      for box, track_id, cls in zip(boxes, ids, clss):
        x1, y1, x2, y2 = map(int, box)
        nome_classe = self.model.names[int(cls)].lower()
        centro = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 1. Agrupamento por Categoria
        if nome_classe in ['person', 'pessoa']:
          pessoas.append({'id': track_id, 'bbox': (x1, y1, x2, y2), 'centro': centro})
        elif nome_classe in ['dog', 'cat', 'cachorro', 'gato', 'animal']:
          animais.append({'id': track_id, 'bbox': (x1, y1, x2, y2), 'nome': nome_classe})
        else:
          # Qualquer outro objeto (saco, garrafa, caixa, objeto_abandonado)
          objetos.append({'id': track_id, 'bbox': (x1, y1, x2, y2), 'centro': centro, 'nome': nome_classe})

    # 2. Desenha Detecções de ANIMAIS (Alerta de Presença)
    for animal in animais:
      x1, y1, x2, y2 = animal['bbox']
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2) # Laranja
      cv2.putText(frame, f"ANIMAL: {animal['nome'].upper()}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # 3. Processa OBJETOS ABANDONADOS
    tempo_atual = time.time()
    for obj in objetos:
      x1, y1, x2, y2 = obj['bbox']
      track_id = obj['id']

      # Verifica se há alguma pessoa próxima (raio de 150 pixels)
      pessoa_proxima = False
      for p in pessoas:
        if self.calcular_distancia(obj['centro'], p['centro']) < 150:
          pessoa_proxima = True
          break

      # Se o objeto está sem ninguém por perto
      if not pessoa_proxima:
        if track_id not in self.tempo_estatico_objetos:
          self.tempo_estatico_objetos[track_id] = tempo_atual
        else:
          tempo_parado = tempo_atual - self.tempo_estatico_objetos[track_id]

          if tempo_parado >= self.limite_segundos:
            # DISPARA ALERTA DE OBJETO ABANDONADO / LIXO
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Vermelho
            cv2.putText(frame, f"ALERTA: DESCARTE ({int(tempo_parado)}s)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
      else:
        # Se uma pessoa se aproximar do objeto novamente, reseta o tempo
        if track_id in self.tempo_estatico_objetos:
          del self.tempo_estatico_objetos[track_id]

    # 4. Desenha PESSOAS
    for p in pessoas:
      x1, y1, x2, y2 = p['bbox']
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Azul
      cv2.putText(frame, "PESSOA", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame, results
