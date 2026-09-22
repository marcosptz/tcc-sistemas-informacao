import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class TrackerComportamental:
    def __init__(self, model_path=None, model_lixo_path=None, model_geral_path='yolo11s.pt', 
                 limite_tempo_estatico_segundos=3, distancia_proximidade_pixels=150):
        """
        Inicializa o rastreador comportamental com inteligência de agentes (Pessoas e Animais)
        e detecção sequencial de descarte irregular.
        """
        path_lixo = model_lixo_path or model_path
        if not path_lixo:
            raise ValueError("É necessário fornecer o caminho do modelo de lixo (model_path ou model_lixo_path).")

        # Seleciona explicitamente GPU CUDA se disponível
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # Inicializa ambos os modelos YOLO
        self.model_lixo = YOLO(path_lixo)
        self.model_geral = YOLO(model_geral_path)

        self.limite_tempo_estatico = limite_tempo_estatico_segundos
        self.distancia_proximidade = distancia_proximidade_pixels

        # Dicionário de estado e histórico de objetos de lixo
        self.historico_objetos = {}
        self.next_synthetic_id = 1000

        # Mapeamento COCO para Agentes Interativos: 0=Pessoa, 15=Gato, 16=Cão
        self.CLASSES_AGENTES = {
            0: 'Pessoa',
            15: 'Gato',
            16: 'Cão'
        }

    def _calcular_centro(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _calcular_distancia(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _distancia_ponto_caixa(self, ponto, box):
        """
        Calcula a menor distância euclidiana entre o centro do objeto e a caixa delimitadora do agente.
        """
        px, py = ponto
        x1, y1, x2, y2 = box
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return math.hypot(dx, dy)

    def _encontrar_ou_criar_id_sintetico(self, centro):
        """
        Associa o objeto a um ID existente por proximidade espacial caso a GPU perca o track_id.
        """
        for tid, estado in self.historico_objetos.items():
            if self._calcular_distancia(centro, estado['centro']) < 50:
                return tid
        self.next_synthetic_id += 1
        return self.next_synthetic_id

    def processar_frame(self, frame):
        """
        Executa o pipeline completo:
        1. Identificação de Pessoas e Animais no frame.
        2. Identificação de Objetos de Lixo.
        3. Associação de Proximidade e Validação Sequencial de Alerta.
        """
        tempo_atual = time.time()
        frame_desenho = frame.copy()

        # -------------------------------------------------------------
        # 1. Rastreamento Geral de Agentes (Pessoas e Animais)
        # -------------------------------------------------------------
        resultados_geral = self.model_geral.track(
            frame, persist=True, verbose=False, conf=0.25, device=self.device
        )
        
        agentes_detectados = []

        if resultados_geral and resultados_geral[0].boxes is not None and len(resultados_geral[0].boxes) > 0:
            boxes_g = resultados_geral[0].boxes.xyxy.cpu().numpy()
            clss_g = resultados_geral[0].boxes.cls.cpu().numpy().astype(int)
            ids_g = resultados_geral[0].boxes.id.cpu().numpy().astype(int) if resultados_geral[0].boxes.id is not None else [None] * len(boxes_g)

            for box, cls_idx, track_id in zip(boxes_g, clss_g, ids_g):
                if cls_idx in self.CLASSES_AGENTES:
                    nome_agente = self.CLASSES_AGENTES[cls_idx]
                    centro = self._calcular_centro(box)
                    
                    agentes_detectados.append({
                        'tipo': nome_agente,
                        'box': box,
                        'centro': centro,
                        'id': track_id
                    })

                    # Desenho do agente (Pessoas = Azul Ciano, Animais = Rosa Magenta)
                    x1, y1, x2, y2 = map(int, box)
                    cor_agente = (255, 200, 0) if cls_idx == 0 else (255, 0, 255)
                    lbl_id = f" #{track_id}" if track_id is not None else ""
                    
                    cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor_agente, 2)
                    cv2.putText(frame_desenho, f"{nome_agente}{lbl_id}", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_agente, 2)

        # -------------------------------------------------------------
        # 2. Rastreamento do Modelo de Lixo
        # -------------------------------------------------------------
        resultados_lixo = self.model_lixo.track(
            frame, persist=True, verbose=False, conf=0.20, imgsz=640, device=self.device
        )
        
        lixo_detectado_no_frame = set()

        if resultados_lixo and resultados_lixo[0].boxes is not None and len(resultados_lixo[0].boxes) > 0:
            boxes_lixo = resultados_lixo[0].boxes.xyxy.cpu().numpy()
            clss_lixo = resultados_lixo[0].boxes.cls.cpu().numpy().astype(int)
            ids_lixo = resultados_lixo[0].boxes.id.cpu().numpy().astype(int) if resultados_lixo[0].boxes.id is not None else None
            nomes_classes_lixo = resultados_lixo[0].names

            for idx, (box, cls_idx) in enumerate(zip(boxes_lixo, clss_lixo)):
                centro_objeto = self._calcular_centro(box)
                nome_obj = nomes_classes_lixo.get(cls_idx, "Lixo")

                # Define ID do rastreamento (da Ultralytics ou Sintético)
                if ids_lixo is not None and idx < len(ids_lixo) and ids_lixo[idx] is not None:
                    track_id = int(ids_lixo[idx])
                else:
                    track_id = self._encontrar_ou_criar_id_sintetico(centro_objeto)

                lixo_detectado_no_frame.add(track_id)

                # Verifica se há Pessoas ou Animais próximos
                agente_proximo = False
                centro_agente_proximo = None
                tipo_agente_proximo = "Agente"

                for agente in agentes_detectados:
                    dist_box = self._distancia_ponto_caixa(centro_objeto, agente['box'])
                    if dist_box <= self.distancia_proximidade:
                        agente_proximo = True
                        centro_agente_proximo = agente['centro']
                        tipo_agente_proximo = agente['tipo']
                        break

                # Inicializa novo objeto no histórico
                if track_id not in self.historico_objetos:
                    self.historico_objetos[track_id] = {
                        'centro': centro_objeto,
                        'box': box,
                        'tempo_criacao': tempo_atual,
                        'tempo_inicio_estatico': tempo_atual,
                        'ultima_vez_visto': tempo_atual,
                        'interacao_detectada': agente_proximo,
                        'alerta': False,
                        'tipo': nome_obj,
                        'agente_tipo': tipo_agente_proximo
                    }

                estado_obj = self.historico_objetos[track_id]
                estado_obj['ultima_vez_visto'] = tempo_atual
                estado_obj['box'] = box

                # Quando um agente (pessoa/animal) está perto do objeto
                if agente_proximo:
                    estado_obj['interacao_detectada'] = True
                    estado_obj['tempo_inicio_estatico'] = tempo_atual
                    estado_obj['agente_tipo'] = tipo_agente_proximo
                    cv2.line(frame_desenho, centro_objeto, centro_agente_proximo, (0, 255, 255), 2, cv2.LINE_AA)

                # Se o objeto se mover (ex: arremessado), atualiza o centro e reseta a contagem
                if self._calcular_distancia(centro_objeto, estado_obj['centro']) > 15:
                    estado_obj['centro'] = centro_objeto
                    estado_obj['tempo_inicio_estatico'] = tempo_atual

                # Avalia Condição de ALERTA DE DESCARTE IRREGULAR:
                # 1. Objeto teve interação com Pessoa/Animal (interacao_detectada = True)
                # 2. O agente se afastou (agente_proximo = False)
                # 3. O objeto ficou estático pelo tempo limite estipulado (ex: 3 segundos)
                tempo_parado = tempo_atual - estado_obj['tempo_inicio_estatico']

                if estado_obj['interacao_detectada'] and not agente_proximo and tempo_parado >= self.limite_tempo_estatico:
                    estado_obj['alerta'] = True

                # Desenho das caixas de estado no frame
                x1, y1, x2, y2 = map(int, box)

                if estado_obj['alerta']:
                    cor_box = (0, 0, 255) # Vermelho
                    lbl_agente = f" ({estado_obj['agente_tipo']})" if estado_obj.get('agente_tipo') else ""
                    label = f"ALERTA: Descarte Irregular #{track_id}{lbl_agente}"
                    cv2.rectangle(frame_desenho, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
                elif agente_proximo:
                    cor_box = (0, 165, 255) # Laranja
                    label = f"{nome_obj} #{track_id} (Presenca: {tipo_agente_proximo})"
                elif estado_obj['interacao_detectada']:
                    cor_box = (0, 255, 255) # Amarelo
                    tempo_restante = max(0, int(self.limite_tempo_estatico - tempo_parado))
                    label = f"{nome_obj} #{track_id} (Aguardando Alerta: {tempo_restante}s)"
                else:
                    cor_box = (128, 128, 128) # Cinza (Objetos de fundo sem interação)
                    label = f"{nome_obj} #{track_id}"

                cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor_box, 2)
                cv2.putText(frame_desenho, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_box, 2)

        # -------------------------------------------------------------
        # 3. Memória de Persistência (caso o YOLO perca o lixo temporariamente)
        # -------------------------------------------------------------
        for track_id, estado_obj in list(self.historico_objetos.items()):
            if track_id not in lixo_detectado_no_frame:
                tempo_sem_ver = tempo_atual - estado_obj['ultima_vez_visto']

                # Mantém na memória por até 5.0 segundos
                if tempo_sem_ver <= 5.0 and estado_obj['interacao_detectada']:
                    agente_proximo = any(
                        self._distancia_ponto_caixa(estado_obj['centro'], ag['box']) <= self.distancia_proximidade 
                        for ag in agentes_detectados
                    )
                    
                    tempo_parado = tempo_atual - estado_obj['tempo_inicio_estatico']
                    if not agente_proximo and tempo_parado >= self.limite_tempo_estatico:
                        estado_obj['alerta'] = True

                    if estado_obj['alerta']:
                        x1, y1, x2, y2 = map(int, estado_obj['box'])
                        lbl_agente = f" ({estado_obj['agente_tipo']})" if estado_obj.get('agente_tipo') else ""
                        cv2.rectangle(frame_desenho, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
                        cv2.putText(frame_desenho, f"ALERTA: Descarte Irregular #{track_id}{lbl_agente}", 
                                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame_desenho, self.historico_objetos
