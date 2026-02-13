#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
import pyaudio
from scipy.fftpack import fft
from scipy.signal import welch, butter, sosfilt
import time
import sys
from collections import deque
from enum import Enum
from threading import Thread
import requests
import json
import threading

# КОНФИГУРАЦИЯ ДЛЯ RASPBERRY PI

# Параметры аудиопотока
BUFFER_SIZE = 2048
SAMPLE_RATE = 44100
FFT_SIZE = 1024
WELCH_SEGMENT = 512
WELCH_OVERLAP = 256

# Интервалы отправки на сервер
SEND_INTERVAL_NORMAL = 2.0      # 2 сек при отсутствии дрона
SEND_INTERVAL_DETECTED = 0.5    # 0.5 сек при обнаружении

# Типы дронов для классификации
class DroneType(Enum):
    UNKNOWN = "Неизвестно"
    SMALL = "Маленький дрон"
    MEDIUM = "Средний дрон"
    LARGE = "Большой дрон"

# Параметры для разных типов дронов
DRONE_PROFILES = {
    DroneType.SMALL: {
        'bands': [(7500, 8000, 0.5), (14500, 15500, 0.5), (0, 500, 1)],
        'fundamental_range': (80, 300),
        'harmonic_count': 4,
        'noise_factor': 0.3
    },
    DroneType.MEDIUM: {
        'bands': [(6500, 7500, 0.4), (12500, 14000, 0.4), (10000, 12000, 0.3), (0, 400, 0.8)],
        'fundamental_range': (60, 200),
        'harmonic_count': 5,
        'noise_factor': 0.25
    },
    DroneType.LARGE: {
        'bands': [(5000, 7000, 0.3), (9500, 11500, 0.3), (13500, 14500, 0.3), (0, 300, 0.6)],
        'fundamental_range': (40, 120),
        'harmonic_count': 6,
        'noise_factor': 0.2
    }
}

# Параметры обработки
HARMONICS_CHECK = True
USE_WELCH = True
ADAPTIVE_THRESHOLDS = True

# Настройки временной фильтрации
HARMONIC_HISTORY_SIZE = 40
CONFIRMATION_THRESHOLD = 0.35  

# Butterworth фильтр для шумоподавления (HIGH-PASS 80Hz)
butter_sos = butter(2, 80, btype='high', fs=SAMPLE_RATE, output='sos')

# Инициализация PyAudio
audio_engine = pyaudio.PyAudio()

# Истории для временной фильтрации
harmonic_history = deque(maxlen=HARMONIC_HISTORY_SIZE)
frequency_history = deque(maxlen=15)
noise_floor_history = deque(maxlen=100)
detected_drone_history = deque(maxlen=20)

# Глобальные переменные для детектора
detected_drone_type = DroneType.UNKNOWN
detection_confidence = 0.0
current_fundamental_freq = None
adaptive_threshold_factor = 1.0
last_detection_time = 0

# Флаги управления
running = True
detection_active = False


# ФУНКЦИИ ОТПРАВКИ НА СЕРВЕР

# КОНФИГУРАЦИЯ СЕРВЕРА
SERVER_URL = "http://192.168.0.223:5000/api/update" 
INSTALLATION_ID = "north"  # ← north, east, south, west

# Переменные для отправки данных
last_sent_status = None
send_lock = threading.Lock()
last_send_time = 0

def send_to_server(detected, frequency, drone_type, confidence): #Отправляет статус на сервер с обработкой ошибок
    global last_sent_status, last_send_time
    
    current_time = time.time()
    
    # Динамический интервал отправки
    send_interval = SEND_INTERVAL_DETECTED if detected else SEND_INTERVAL_NORMAL
    
    # Отправляем только если изменился статус или прошло достаточно времени
    if (detected != last_sent_status or current_time - last_send_time > send_interval):
        try:
            payload = {
                "installation_id": INSTALLATION_ID,
                "detected": detected,
                "frequency": float(frequency) if frequency else 0.0,
                "confidence": float(confidence) if confidence else 0.0,
                "timestamp": time.time()
            }
            
            response = requests.post(
                SERVER_URL,
                json=payload,
                timeout=3
            )
            
            if response.status_code == 200:
                print(f"✓ Сервер: обнаружение={detected}")
                last_sent_status = detected
                last_send_time = current_time
            else:
                print(f"✗ Ошибка отправки: HTTP {response.status_code}")
        
        except requests.exceptions.ConnectTimeout:
            print(f"✗ Таймаут подключения")
        except requests.exceptions.ReadTimeout:
            print(f"✗ Таймаут чтения")
            last_sent_status = detected
            last_send_time = current_time
        except requests.exceptions.ConnectionError:
            print(f"✗ Ошибка сети: проверьте IP сервера")
        except Exception as e:
            print(f"✗ Ошибка: {type(e).__name__}")

# ФУНКЦИИ ОБРАБОТКИ СПЕКТРА

def compute_welch_spectrum(audio_data, fs, nperseg=512, noverlap=256): #Вычисляет спектр методом Уэлча
    f, Pxx = welch(audio_data, fs=fs, nperseg=nperseg,
                   noverlap=noverlap, nfft=FFT_SIZE,
                   scaling='spectrum', window='hann')
    magnitude = np.sqrt(Pxx)
    magnitude_log = np.log1p(magnitude * 1000)
    return f, magnitude_log[:len(f)]

def update_noise_floor(spectrum, freq_axis): #Обновляет уровень фонового шума
    high_freq_mask = freq_axis > 10000
    if np.any(high_freq_mask):
        noise_level = np.median(spectrum[high_freq_mask])
        noise_floor_history.append(noise_level)
        return noise_level
    return 0.0

def get_adaptive_threshold(base_threshold): #Возвращает порог с учётом уровня шума
    if not ADAPTIVE_THRESHOLDS or len(noise_floor_history) < 10:
        return base_threshold
    
    current_noise = np.median(list(noise_floor_history)[-10:])
    adaptive_factor = 1.0 + (current_noise * 0.5)
    return base_threshold * adaptive_factor

def classify_drone_type(peak_freqs, peak_vals, fundamental_freq, spectrum, freq_axis): #Классифицирует тип дрона по спектральным характеристикам
    scores = {}
    
    for drone_type, profile in DRONE_PROFILES.items():
        score = 0
        
        # Проверка основной частоты
        freq_low, freq_high = profile['fundamental_range']
        if freq_low <= fundamental_freq <= freq_high:
            score += 0.3
        
        # Проверка характерных полос
        band_score = 0
        for band_low, band_high, _ in profile['bands']:
            band_mask = (freq_axis >= band_low) & (freq_axis <= band_high)
            if np.any(band_mask):
                band_power = np.mean(spectrum[band_mask])
                if band_power > get_adaptive_threshold(0.2):
                    band_score += 1
        
        score += (band_score / len(profile['bands'])) * 0.4
        
        # Проверка количества пиков
        if len(peak_freqs) >= profile['harmonic_count']:
            score += 0.3
        
        scores[drone_type] = score
    
    # Выбираем тип с наибольшим score
    best_type = max(scores, key=scores.get)
    if scores[best_type] > 0.5:
        return best_type
    
    return DroneType.UNKNOWN

def calculate_confidence(harmonics_found, total_harmonics, fundamental_amp, drone_type): #Вычисляет уверенность обнаружения
    harmonic_ratio = harmonics_found / total_harmonics if total_harmonics > 0 else 0
    confidence = harmonic_ratio * 0.6
    
    amp_factor = min(fundamental_amp / 1.0, 1.0)
    confidence += amp_factor * 0.2
    
    # Учитываем стабильность частоты
    if len(frequency_history) >= 3:
        recent = list(frequency_history)[-3:]
        if np.std(recent) < 50:
            confidence += 0.2
    
    return min(confidence, 1.0)

def check_harmonics_with_delay_enhanced(frequency_axis, magnitude_log): #проверка гармоник с классификацией типа дрона
    global detected_drone_type, detection_confidence, adaptive_threshold_factor
    
    # 1. Предварительная фильтрация
    overall_volume = np.max(magnitude_log)
    if overall_volume < get_adaptive_threshold(0.3):  
        harmonic_history.append(False)
        return False, None, DroneType.UNKNOWN, 0.0
    
    # 2. Обновляем уровень шума
    update_noise_floor(magnitude_log, frequency_axis)
    
    # 3. Находим пики с адаптивным порогом
    peak_indices = []
    peak_values = []
    peak_frequencies = []
    
    adaptive_peak_threshold = get_adaptive_threshold(0.4)  
    
    for i in range(10, len(magnitude_log) - 10):
        if magnitude_log[i] > adaptive_peak_threshold:
            # Проверяем, что это пик
            is_peak = True
            for offset in range(1, 11):  
                if magnitude_log[i] <= magnitude_log[i - offset] or \
                   magnitude_log[i] <= magnitude_log[i + offset]:
                    is_peak = False
                    break
            
            if is_peak and magnitude_log[i] > adaptive_peak_threshold * 1.2:  
                peak_indices.append(i)
                peak_values.append(magnitude_log[i])
                peak_frequencies.append(frequency_axis[i])
    
    # Если слишком мало пиков
    if len(peak_indices) < 3:  
        harmonic_history.append(False)
        return False, None, DroneType.UNKNOWN, 0.0
    
    # 4. Находим основную частоту
    fundamental_freq = None
    fundamental_val = 0
    fundamental_idx = -1
    
    for idx, freq in enumerate(peak_frequencies):
        if 30 < freq < 1000:
            if peak_values[idx] > fundamental_val:
                fundamental_val = peak_values[idx]
                fundamental_freq = freq
                fundamental_idx = idx
    
    if fundamental_freq is None or fundamental_freq < 30:
        harmonic_history.append(False)
        return False, None, DroneType.UNKNOWN, 0.0
    
    # 5. Определяем тип дрона
    drone_type = classify_drone_type(peak_frequencies, peak_values, fundamental_freq,
                                     magnitude_log, frequency_axis)
    
    # 6. Проверяем гармоники
    profile = DRONE_PROFILES.get(drone_type, DRONE_PROFILES[DroneType.SMALL])
    harmonic_ratios = list(range(2, profile['harmonic_count'] + 2))
    
    harmonics_found = 0
    harmonic_matches = []
    
    # Нормализуем амплитуды
    normalized_peaks = []
    for i in range(len(peak_frequencies)):
        normalized = peak_values[i] / fundamental_val if fundamental_val > 0 else 0
        normalized_peaks.append((peak_frequencies[i], normalized, peak_values[i]))
    
    for ratio in harmonic_ratios:
        target_freq = fundamental_freq * ratio
        
        # Ищем ближайший пик
        closest_peak = None
        min_diff = float('inf')
        
        for freq, norm, val in normalized_peaks:
            if freq < 50:
                continue
            
            freq_diff = abs(freq - target_freq)
            relative_diff = freq_diff / target_freq
            
            tolerance = 0.05 if ratio <= 4 else 0.07 
            
            if relative_diff < tolerance and freq_diff < min_diff:
                min_diff = freq_diff
                closest_peak = (freq, norm, val)
        
        if closest_peak and closest_peak[1] > 0.15: 
            harmonics_found += 1
            harmonic_matches.append((ratio, closest_peak[0], closest_peak[1]))
    
    # 7. Вычисляем уверенность
    confidence = calculate_confidence(harmonics_found, len(harmonic_ratios),
                                     fundamental_val, drone_type)
    
    # 8. Критерии обнаружения
    current_result = False
    min_harmonics_required = 2 if drone_type == DroneType.SMALL else 1 
    
    if (harmonics_found >= min_harmonics_required and
        fundamental_val > get_adaptive_threshold(0.5)): 
        
        # Проверяем стабильность частоты
        if len(frequency_history) >= 5:
            recent_freqs = list(frequency_history)[-5:]
            freq_std = np.std(recent_freqs)
            if freq_std < 100:
                current_result = True
        elif len(frequency_history) > 0:
            current_result = True
    
    # Сохраняем частоту в историю
    frequency_history.append(fundamental_freq)
    harmonic_history.append(current_result)
    
    # Временная фильтрация
    if len(harmonic_history) < HARMONIC_HISTORY_SIZE // 2: 
        return False, fundamental_freq, drone_type, confidence
    
    positive_count = sum(harmonic_history)
    total_count = len(harmonic_history)
    positive_ratio = positive_count / total_count if total_count > 0 else 0
    
    if positive_ratio >= CONFIRMATION_THRESHOLD:
        if frequency_history:
            confirmed_frequency = np.median(list(frequency_history)[-10:])
        else:
            confirmed_frequency = fundamental_freq
        
        detected_drone_type = drone_type
        detection_confidence = confidence
        return True, confirmed_frequency, drone_type, confidence
    
    return False, fundamental_freq, drone_type, confidence

# ФУНКЦИИ МИКРОФОНА

def get_available_microphones(): #Получает список доступных микрофонов
    microphones = []
    info = audio_engine.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    for i in range(num_devices):
        device_info = audio_engine.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            device_name = device_info.get('name')
            microphones.append((i, device_name))
    
    return microphones

def create_stream(buffer, rate, device_index=None): #Инициализация аудиопотока
    try:
        stream = audio_engine.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=buffer,
            stream_callback=None
        )
        return stream
    except Exception as e:
        print(f"Ошибка создания потока: {e}")
        return None

def safe_exit(): #Безопасно закрывает программу
    global running
    running = False
    
    print("\n\nЗакрываю программу...")
    
    try:
        if 'audio_stream' in globals() and audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
            print("Аудиопоток закрыт")
    except Exception as e:
        print(f"Ошибка при закрытии потока: {e}")
    
    try:
        if 'audio_engine' in globals():
            audio_engine.terminate()
            print("Аудиодвижок закрыт")
    except Exception as e:
        print(f"Ошибка при закрытии движка: {e}")
    
    sys.exit(0)

# ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ АУДИО

def audio_processing_thread(): #Поток обработки аудио
    global running, detection_active, last_detection_time
    
    print("\n" + "="*70)
    print("ДЕТЕКТОР ДРОНОВ - HYBRID VERSION")
    print("="*70)
    print(f"Установка: {INSTALLATION_ID.upper()}")
    print(f"Сервер: {SERVER_URL}")
    print(f"Параметры: SR={SAMPLE_RATE}Hz, FFT={FFT_SIZE}, Welch={WELCH_SEGMENT}/{WELCH_OVERLAP}")
    print(f"Алгоритм")
    print("="*70 + "\n")
    
    frame_count = 0
    detection_history_text = ""
    
    while running:
        try:
            # Читаем аудиоданные
            raw_audio = np.frombuffer(
                audio_stream.read(BUFFER_SIZE, exception_on_overflow=False),
                dtype=np.int16
            ).astype(np.float32)
            
            # Нормализация
            raw_audio = raw_audio / 32768.0
            
            # HIGH-PASS фильтрация
            filtered = sosfilt(butter_sos, raw_audio)
            
            # Вычисляем спектр
            frequency_axis, magnitude_log = compute_welch_spectrum(
                filtered, SAMPLE_RATE, WELCH_SEGMENT, WELCH_OVERLAP
            )
            
            # Проверяем гармоники
            harmonics_confirmed, current_freq, drone_type, confidence = \
                check_harmonics_with_delay_enhanced(frequency_axis, magnitude_log)
            
            # Обновляем статус детекции
            if harmonics_confirmed and confidence > 0.5:
                detection_active = True
                last_detection_time = time.time()
                indicator_color = "🟢"
            else:
                # Держим детекцию активной 0.5 сек после последнего сигнала
                if time.time() - last_detection_time > 0.5:
                    detection_active = False
                indicator_color = "🟢" if detection_active else "🔴"
            
            # ОТПРАВЛЯЕМ ДАННЫЕ НА СЕРВЕР
            send_to_server(detection_active, current_freq, drone_type.value, confidence)
            
            # Формируем строку статуса
            frame_count += 1
            if frame_count % 10 == 0:  # Обновляем каждые 10 кадров (~500ms)
                freq_str = f"{current_freq:.0f}" if current_freq else "---"
                conf_str = f"{confidence:.0%}" if confidence > 0 else "---"
                type_str = drone_type.value if drone_type != DroneType.UNKNOWN else "Неизвестно"
                
                # История: + если обнаружение, - если нет
                if len(harmonic_history) > 0:
                    detection_history_text = "".join(
                        ["+" if h else "-" for h in list(harmonic_history)[-20:]]
                    )
                
                print(
                    f"\r{indicator_color} F:{freq_str:>6}Hz | "
                    f"Conf:{conf_str:>4} | Type:{type_str:15} | "
                    f"Status:{'DETECTED' if detection_active else 'CLEAR':10} | "
                    f"История: {detection_history_text:20}",
                    end='', flush=True
                )
            
            # time.sleep для синхронизации
            time.sleep(0.001)
        
        except IOError as e:
            print(f"\nОшибка буфера: {e}")
            time.sleep(0.1)
        except Exception as e:
            print(f"\nОшибка обработки: {e}")
            time.sleep(0.1)


# ГЛАВНАЯ ПРОГРАММА

if __name__ == "__main__":
    try:
        # Получаем микрофоны
        microphones = get_available_microphones()
        if not microphones:
            print("⚠️ Микрофоны не найдены! Проверьте подключение.")
            sys.exit(1)
        
        # Используем первый микрофон
        current_device_index = microphones[0][0]
        print(f"Используется микрофон: {microphones[0][1]}")
        
        # Создаем аудиопоток
        audio_stream = create_stream(BUFFER_SIZE, SAMPLE_RATE, current_device_index)
        if audio_stream is None:
            print("Не удалось создать аудиопоток!")
            sys.exit(1)
        
        # Запускаем поток обработки аудио
        audio_thread = Thread(target=audio_processing_thread, daemon=True)
        audio_thread.start()
        
        print(f"\nДетектор запущен (установка: {INSTALLATION_ID.upper()})")
        print("Отправка данных на сервер активна")
        print("Нажмите Ctrl+C для остановки\n")
        
        # Держим основной поток активным
        while running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка: {e}")
    finally:
        safe_exit()
