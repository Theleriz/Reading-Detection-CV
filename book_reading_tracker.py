"""
Трекер чтения книги — MediaPipe Tasks API (новая нотация)
Детектирует чтение по наклону головы (нос вниз) и направлению взгляда.

Установка:
    pip install opencv-contrib-python mediapipe numpy

Скачать модель face_landmarker:
    curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    # или укажи свой путь в MODEL_PATH ниже

Запуск:
    python book_reading_tracker.py
"""

import os
import time
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"   # путь к .task-файлу модели

HEAD_TILT_THRESHOLD = 0.02   # порог наклона головы (уменьши = чувствительнее)
GAZE_THRESHOLD      = 0.05   # порог взгляда вниз

# Landmark indices 478 points
NOSE_TIP         = 1
FOREHEAD         = 10
CHIN             = 152
LEFT_IRIS        = 468
RIGHT_IRIS       = 473
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374

def get_head_tilt(landmarks: list) -> float:
    """
    Нормализованный наклон головы вниз.
    > 0  ->  голова опущена,  0  ->  прямо.
    """
    nose_y     = landmarks[NOSE_TIP].y
    forehead_y = landmarks[FOREHEAD].y
    chin_y     = landmarks[CHIN].y

    face_height = chin_y - forehead_y
    if face_height == 0:
        return 0.0

    mid_y = (forehead_y + chin_y) / 2
    return (nose_y - mid_y) / face_height


def get_gaze_vertical(landmarks: list) -> float:
    """
    Нормализованное положение радужки в веке.
    > 0  ->  взгляд вниз,  < 0  ->  вверх.
    """
    values = []
    for iris_idx, top_idx, bot_idx in [
        (LEFT_IRIS,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM),
        (RIGHT_IRIS, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    ]:
        iris_y = landmarks[iris_idx].y
        top_y  = landmarks[top_idx].y
        bot_y  = landmarks[bot_idx].y
        height = bot_y - top_y
        if height == 0:
            continue
        values.append((iris_y - top_y) / height - 0.5)

    return float(np.mean(values)) if values else 0.0


def check_reading(head_tilt: float, gaze: float) -> bool:
    # Weighted score — head tilt counts more, gaze is supporting signal
    score = (head_tilt / HEAD_TILT_THRESHOLD) * 0.6 + (gaze / GAZE_THRESHOLD) * 0.5
    return score > 0.5


def draw_hud(frame: np.ndarray,
             reading: bool,
             head_tilt: float,
             gaze: float,
             read_sec: float,
             total_sec: float) -> None:

    h, w = frame.shape[:2]
    color = (0, 220, 80) if reading else (30, 80, 220)

    # FIX: ограничиваем размер панели реальными размерами кадра,
    # чтобы не упасть с ошибкой на маленьких разрешениях
    ph = min(165, h)
    pw = min(330, w)

    # Полупрозрачный фон панели.
    # roi — это VIEW (не копия) на frame, поэтому addWeighted с dst=roi
    # пишет результат прямо в frame; отдельное присваивание не нужно.
    roi = frame[0:ph, 0:pw]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)

    label = "READING" if reading else "NOT READING"
    cv2.putText(frame, label, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"HEAD ANGLE: {head_tilt:+.3f}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"EYE ANGLE: {gaze:+.3f}", (10, 93),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    mr, sr = divmod(int(read_sec), 60)
    mt, st = divmod(int(total_sec), 60)
    cv2.putText(frame, f"READING: {mr:02d}:{sr:02d}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"SESSION: {mt:02d}:{st:02d}", (10, 153),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    # Цветной индикатор в углу
    cv2.circle(frame, (w - 28, 28), 20, color, -1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Основной код
# ──────────────────────────────────────────────

def main() -> None:
    # FIX: проверяем наличие файла модели до создания детектора,
    # чтобы получить понятное сообщение об ошибке вместо исключения MediaPipe
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: модель не найдена по пути '{MODEL_PATH}'")
        print("Скачай командой:")
        print("  curl -O https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        return

    # --- Новая Tasks API: создаём опции через dataclass ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,      # VIDEO = синхронный покадровый режим
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера недоступна.")
        return

    read_sec   = 0.0
    sess_start = time.time()
    prev_time  = sess_start
    reading    = False

    print("=" * 50)
    print("  Трекер чтения (MediaPipe Tasks API)")
    print("  Нажмите Q для выхода")
    print("=" * 50)

    # --- Создаём детектор через фабричный метод Tasks API ---
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now       = time.time()
            dt        = now - prev_time
            prev_time = now
            total_sec = now - sess_start

            head_tilt = 0.0
            gaze      = 0.0

            # --- Конвертируем кадр в mp.Image (новый тип Tasks API) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # FIX: timestamp считается от старта сессии, а не от эпохи —
            # гарантированно монотонно возрастает; VIDEO-режим MediaPipe
            # молча игнорирует кадры с неубывающим timestamp_ms
            timestamp_ms = int((now - sess_start) * 1000)

            # --- detect_for_video вместо face_mesh.process ---
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                # result.face_landmarks[0] — список NormalizedLandmark
                landmarks = result.face_landmarks[0]

                head_tilt = get_head_tilt(landmarks)
                gaze      = get_gaze_vertical(landmarks)
                reading   = check_reading(head_tilt, gaze)

                if reading:
                    read_sec += dt

                # Tasks API не включает draw_landmarks — рисуем вручную
                h_f, w_f = frame.shape[:2]
                for lm in landmarks:
                    cx = int(lm.x * w_f)
                    cy = int(lm.y * h_f)
                    cv2.circle(frame, (cx, cy), 1, (80, 180, 80), -1)
            else:
                reading = False

            draw_hud(frame, reading, head_tilt, gaze, read_sec, total_sec)
            cv2.imshow("Book Reading Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    total = time.time() - sess_start
    mr, sr = divmod(int(read_sec), 60)
    mt, st = divmod(int(total), 60)
    pct = read_sec / total * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print(f"  Время чтения : {mr:02d}:{sr:02d}")
    print(f"  Общее время  : {mt:02d}:{st:02d}")
    print(f"  % чтения     : {pct:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()