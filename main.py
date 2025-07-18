import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Inicializar o modelo YOLOv8
model = YOLO("yolov8n.pt")  # Substituir pelo seu modelo personalizado se necessário

# Inicializar o rastreador SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Contadores
total_counter = 0
carros = 0
motas = 0
camioes = 0
pessoas = 0

# IDs rastreados para evitar contagem duplicada
ids_detectados = set()

# Classes de interesse no COCO
CLASSES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Captura do vídeo
cap = cv2.VideoCapture("video1.mp4")

# Obter dimensões do frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Criar o gravador de vídeo
output = cv2.VideoWriter("saida_detectada.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Posição da linha de contagem
line_position = int(frame_height * 0.55)
line_color = (0, 255, 0)  # Verde

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção com YOLOv8
    results = model(frame, verbose=False)[0]

    detections = []

    for result in results.boxes:
        class_id = int(result.cls[0])
        conf = float(result.conf[0])
        if class_id in CLASSES and conf > 0.4:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            detections.append([x1, y1, x2, y2, conf, class_id])

    # Preparar detecções para o SORT (sem class_id)
    dets = np.array([[*det[:5]] for det in detections]) if detections else np.empty((0, 5))

    # Atualizar o rastreador
    tracked_objects = tracker.update(dets)

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Desenhar caixa e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Verificar se cruzou a linha e ainda não foi contado
        if line_position - 5 < cy < line_position + 5 and track_id not in ids_detectados:
            ids_detectados.add(track_id)
            total_counter += 1

            # Verificar classe associada ao objeto rastreado
            for det in detections:
                dx1, dy1, dx2, dy2, _, class_id = det
                if abs(cx - (dx1 + dx2) / 2) < 20 and abs(cy - (dy1 + dy2) / 2) < 20:
                    if class_id == 2:
                        carros += 1
                    elif class_id == 3:
                        motas += 1
                    elif class_id == 5 or class_id == 7:
                        camioes += 1
                    elif class_id == 0:
                        pessoas += 1
                    break

    # Linha de contagem
    cv2.line(frame, (0, line_position), (frame_width, line_position), line_color, 2)

    # Mostrar contagens no frame
    cv2.putText(frame, f"Total de Veiculos: {total_counter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(frame, f"Carros: {carros}  Motas: {motas}  Camioes: {camioes}  Pessoas: {pessoas}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Mostrar frame e gravar
    cv2.imshow("Deteccao de Trafego com YOLOv8 + SORT", frame)
    output.write(frame)

    # Tecla para sair
    if cv2.waitKey(1) == ord('q'):
        break

# Libertar recursos
cap.release()
output.release()
cv2.destroyAllWindows()
