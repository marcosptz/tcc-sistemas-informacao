import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class TrackerComportamental:
    def __init__(self, model_path=None, model_lixo_path=None, model_geral_path='yolo11s.pt', 
                 limite_tempo_estatico_segundos=3, distancia_proximidade_pixels=100):
        """
        Inicializa o rastreador comportamental otimizado com prevenção de falsos positivos
        e memória de persistência para objetos recém-descartados.
        """
        path_lixo = model_lixo_path or model_path
        if not path_lixo:
            raise ValueError("É necessário fornecer o caminho do modelo de lixo (model_path ou model_lixo_path).")

        # Seleciona explicitamente a GPU CUDA se disponível
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # Inicializa ambos os modelos da Ultralytics
        self.model_lixo = YOLO(path_lixo)
        self.model_geral = YOLO(model_geral_path)

        self.limite_tempo_estatico = limite_tempo_estatico_segundos
        self.distancia_proximidade = distancia_proximidade_pixels

        # Dicionário de estado dos objetos/animais rastreados
        self.historico_objetos = {}

        # Mapeamento de classes do modelo COCO para Pessoas e Animais
        self.CLASSES_GERAIS_INTERESSE = {
            0: 'Pessoa',
            16: 'Cão',
            17: 'Gato'
        }

    def _calcular_centro(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _calcular_distancia(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def processar_frame(self, frame):
        """
        Processa um frame individual com inteligência contra objetos pré-existentes e memória de rastreio.
        """
        tempo_atual = time.time()
        frame_desenho = frame.copy()

        # -------------------------------------------------------------
        # 1. Executa Rastreamento Geral (Pessoas e Animais) na GPU
        # -------------------------------------------------------------
        resultados_geral = self.model_geral.track(
            frame, persist=True, verbose=False, conf=0.25, device=self.device
        )
        
        centros_pessoas = []
        centros_animais = []

        if resultados_geral and resultados_geral[0].boxes is not None and resultados_geral[0].boxes.id is not None:
            boxes_geral = resultados_geral[0].boxes.xyxy.cpu().numpy()
            clss_geral = resultados_geral[0].boxes.cls.cpu().numpy().astype(int)
            ids_geral = resultados_geral[0].boxes.id.cpu().numpy().astype(int)

            for box, cls_idx, track_id in zip(boxes_geral, clss_geral, ids_geral):
                if cls_idx in self.CLASSES_GERAIS_INTERESSE:
                    centro = self._calcular_centro(box)
                    nome_classe = self.CLASSES_GERAIS_INTERESSE[cls_idx]

                    if cls_idx == 0:  # Pessoa
                        centros_pessoas.append(centro)
                        cor = (255, 200, 0)
                    else:  # Animais
                        centros_animais.append({'id': track_id, 'centro': centro, 'box': box, 'tipo': nome_classe})
                        cor = (255, 150, 50)

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(frame_desenho, f"{nome_classe} #{track_id}", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        # -------------------------------------------------------------
        # 2. Executa Rastreamento do Modelo de Lixo na GPU
        # -------------------------------------------------------------
        resultados_lixo = self.model_lixo.track(
            frame, persist=True, verbose=False, conf=0.10, imgsz=640, device=self.device
        )
        
        lixo_detectado_no_frame = set()

        if resultados_lixo and resultados_lixo[0].boxes is not None and resultados_lixo[0].boxes.id is not None:
            boxes_lixo = resultados_lixo[0].boxes.xyxy.cpu().numpy()
            clss_lixo = resultados_lixo[0].boxes.cls.cpu().numpy().astype(int)
            ids_lixo = resultados_lixo[0].boxes.id.cpu().numpy().astype(int)
            nomes_classes_lixo = resultados_lixo[0].names

            for box, cls_idx, track_id in zip(boxes_lixo, clss_lixo, ids_lixo):
                centro_objeto = self._calcular_centro(box)
                nome_obj = nomes_classes_lixo.get(cls_idx, "Lixo")
                lixo_detectado_no_frame.add(track_id)

                # Inicialização do Objeto
                if track_id not in self.historico_objetos:
                    # Se não havia nenhuma pessoa perto no momento em que o objeto apareceu no vídeo,
                    # ele é classificado como "Pré-existente" para evitar alertas falsos quando alguém passar perto.
                    pessoa_presente_no_surgimento = any(
                        self._calcular_distancia(centro_objeto, cp) <= self.distancia_proximidade 
                        for cp in centros_pessoas
                    )
                    
                    self.historico_objetos[track_id] = {
                        'centro': centro_objeto,
                        'box': box,
                        'tempo_criacao': tempo_atual,
                        'tempo_inicio_estatico': tempo_atual,
                        'ultima_vez_visto': tempo_atual,
                        'interacao_humana': False,
                        'preexistente': not pessoa_presente_no_surgimento,
                        'alerta': False,
                        'tipo': nome_obj
                    }
                
                estado_obj = self.historico_objetos[track_id]
                estado_obj['ultima_vez_visto'] = tempo_atual
                estado_obj['box'] = box

                # Verifica Proximidade de Pessoas
                pessoa_proxima = False
                for centro_pessoa in centros_pessoas:
                    distancia = self._calcular_distancia(centro_objeto, centro_pessoa)
                    if distancia <= self.distancia_proximidade:
                        pessoa_proxima = True
                        # Apenas marca interação humana se O OBJETO NÃO FOR PRÉ-EXISTENTE
                        if not estado_obj['preexistente']:
                            estado_obj['interacao_humana'] = True
                            estado_obj['tempo_inicio_estatico'] = tempo_atual
                        cv2.line(frame_desenho, centro_objeto, centro_pessoa, (0, 255, 255), 1, cv2.LINE_AA)
                        break

                if self._calcular_distancia(centro_objeto, estado_obj['centro']) > 15:
                    estado_obj['centro'] = centro_objeto
                    estado_obj['tempo_inicio_estatico'] = tempo_atual

                # Avalia Condição de Descarte Irregular
                tempo_parado = tempo_atual - estado_obj['tempo_inicio_estatico']

                if estado_obj['interacao_humana'] and not pessoa_proxima and tempo_parado >= self.limite_tempo_estatico:
                    estado_obj['alerta'] = True

                # Desenho do Bounding Box do Lixo
                x1, y1, x2, y2 = map(int, box)
                if estado_obj['alerta']:
                    cor_box = (0, 0, 255)
                    label = f"ALERTA: Descarte Irregular #{track_id}"
                    cv2.rectangle(frame_desenho, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 3)
                elif estado_obj['preexistente']:
                    cor_box = (100, 100, 100)
                    label = f"{nome_obj} #{track_id} (Pre-existente)"
                elif estado_obj['interacao_humana']:
                    cor_box = (0, 165, 255)
                    label = f"{nome_obj} #{track_id} (Aguardando {int(tempo_parado)}s)"
                else:
                    cor_box = (128, 128, 128)
                    label = f"{nome_obj} #{track_id}"

                cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor_box, 2)
                cv2.putText(frame_desenho, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_box, 2)

        # -------------------------------------------------------------
        # 3. Memória de Persistência (Desenha objetos em alerta se o YOLO perder a detecção)
        # -------------------------------------------------------------
        for track_id, estado_obj in list(self.historico_objetos.items()):
            if isinstance(track_id, str) and track_id.startswith("animal_"):
                continue

            # Se o objeto não foi detectado no frame atual, mas estava em interação ou alerta recente
            if track_id not in lixo_detectado_no_frame:
                tempo_sem_ver = tempo_atual - estado_obj['ultima_vez_visto']

                # Mantém na memória por até 5 segundos após sumir da detecção do YOLO
                if tempo_sem_ver <= 5.0 and estado_obj['interacao_humana']:
                    pessoa_proxima = any(
                        self._calcular_distancia(estado_obj['centro'], cp) <= self.distancia_proximidade 
                        for cp in centros_pessoas
                    )
                    
                    tempo_parado = tempo_atual - estado_obj['tempo_inicio_estatico']
                    if not pessoa_proxima and tempo_parado >= self.limite_tempo_estatico:
                        estado_obj['alerta'] = True

                    if estado_obj['alerta']:
                        x1, y1, x2, y2 = map(int, estado_obj['box'])
                        cv2.rectangle(frame_desenho, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 3)
                        cv2.putText(frame_desenho, f"ALERTA: Descarte Irregular #{track_id}", 
                                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # -------------------------------------------------------------
        # 4. Avalia Abandono de Animais
        # -------------------------------------------------------------
        for animal in centros_animais:
            track_id = f"animal_{animal['id']}"
            centro_animal = animal['centro']
            
            if track_id not in self.historico_objetos:
                self.historico_objetos[track_id] = {
                    'centro': centro_animal,
                    'tempo_inicio_estatico': tempo_atual,
                    'interacao_humana': False,
                    'alerta': False,
                    'tipo': animal['tipo']
                }

            estado_animal = self.historico_objetos[track_id]

            pessoa_proxima = False
            for centro_pessoa in centros_pessoas:
                if self._calcular_distancia(centro_animal, centro_pessoa) <= self.distancia_proximidade:
                    pessoa_proxima = True
                    estado_animal['interacao_humana'] = True
                    estado_animal['tempo_inicio_estatico'] = tempo_atual
                    break

            tempo_parado_animal = tempo_atual - estado_animal['tempo_inicio_estatico']

            if estado_animal['interacao_humana'] and not pessoa_proxima and tempo_parado_animal >= (self.limite_tempo_estatico * 2):
                estado_animal['alerta'] = True
                x1, y1, x2, y2 = map(int, animal['box'])
                cv2.putText(frame_desenho, f"ALERTA: Possivel Abandono ({animal['tipo']})", 
                            (x1, max(y1 - 25, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame_desenho, self.historico_objetos
