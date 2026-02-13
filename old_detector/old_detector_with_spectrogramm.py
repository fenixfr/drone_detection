#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
import sys
from collections import deque


# Пороги обнаружения сигнала
banda_1_start, banda_1_end, amplitud_1 = 7500, 8000, 0.5
banda_2_start, banda_2_end, amplitud_2 = 15000, 15500, 0.5
banda_3_start, banda_3_end, amplitud_3 = 0, 500, 1

# Дополнительные параметры для проверки
HARMONICS_CHECK = True  # Включить проверку гармоник
MIN_HARMONIC_PEAKS = 4  # Минимальное количество пиков
PEAK_THRESHOLD = 0.6   # Повышаем порог для пиков

# Настройки временной фильтрации гармоник
HARMONIC_HISTORY_SIZE = 30  # Увеличиваем историю
CONFIRMATION_THRESHOLD = 0.4  # 40% положительных проверок для подтверждения

bands_detected_history = deque(maxlen=60)

# Параметры аудиопотока
BUFFER_SIZE = 4048
SAMPLE_RATE = 44100
FFT_SIZE = 1024

# Инициализация
audio_engine = pyaudio.PyAudio()
window = tk.Tk()
window.title("Спектральный анализатор частот - Реальное время")
window.configure(bg='#d0d0d0')  # СЕРЫЙ ФОН

# Получение списка доступных микрофонов
def get_available_microphones():
    microphones = []
    info = audio_engine.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    for i in range(num_devices):
        device_info = audio_engine.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:  # Проверяем, что устройство является входным (микрофоном)
            device_name = device_info.get('name')
            microphones.append((i, device_name))
    
    return microphones

# Получаем список микрофонов
microphones = get_available_microphones()
current_device_index = 0  # По умолчанию первый микрофон

# История проверок гармоник для временной фильтрации
harmonic_history = deque(maxlen=HARMONIC_HISTORY_SIZE)
confirmed_frequency = None  # Сохраняем подтвержденную основную частоту
frequency_history = deque(maxlen=10)  # История основных частот

# Функция для дополнительной проверки гармоник с временной фильтрацией
def check_harmonics_with_delay(frequency_axis, magnitude_log):
    if not HARMONICS_CHECK:
        return True, None
    
    # 1. Предварительная фильтрация - убираем слишком тихие сигналы
    overall_volume = np.max(magnitude_log)
    if overall_volume < 0.3:  # Если общий уровень сигнала слишком низкий
        harmonic_history.append(False)
        return False, None
    
    # 2. Находим все пики выше порога
    peak_indices = []
    peak_values = []
    peak_frequencies = []
    
    # Поиск пиков: смотрим на окрестность 5 точек
    for i in range(10, len(magnitude_log) - 10):
        if magnitude_log[i] > PEAK_THRESHOLD:
            # Проверяем, что это действительно пик в окрестности ±10 точек
            is_peak = True
            for offset in range(1, 11):
                if magnitude_log[i] <= magnitude_log[i - offset] or magnitude_log[i] <= magnitude_log[i + offset]:
                    is_peak = False
                    break
            
            if is_peak and magnitude_log[i] > 0.5:  # Дополнительный порог для пиков
                peak_indices.append(i)
                peak_values.append(magnitude_log[i])
                peak_frequencies.append(frequency_axis[i])
    
    # Если пиков меньше 4, скорее всего это не гармоники дрона
    if len(peak_indices) < 4:
        harmonic_history.append(False)
        return False, None
    
    # 3. Находим самую сильную низкочастотную компоненту как основную
    # Сначала сортируем по амплитуде
    sorted_by_amplitude = sorted(zip(peak_values, peak_frequencies, peak_indices), 
                                 key=lambda x: x[0], reverse=True)
    
    # Берем 5 самых сильных пиков
    top_peaks = sorted_by_amplitude[:min(5, len(sorted_by_amplitude))]
    
    # Ищем среди них самую низкочастотную как основную
    fundamental_freq = None
    fundamental_val = 0
    
    for val, freq, idx in top_peaks:
        if freq < 2000:  # Основная частота дрона обычно ниже 2 кГц
            if val > fundamental_val:
                fundamental_val = val
                fundamental_freq = freq
            break
    
    if fundamental_freq is None or fundamental_freq < 50:  # Слишком низкая частота - шум
        harmonic_history.append(False)
        return False, None
    
    # Сохраняем частоту в историю
    frequency_history.append(fundamental_freq)
    
    # 4. Новый алгоритм проверки гармоник - ищем кратные частоты
    harmonics_found = 0
    harmonic_ratios = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # Сначала нормализуем амплитуды относительно основной
    normalized_peaks = []
    for i in range(len(peak_frequencies)):
        normalized = peak_values[i] / fundamental_val if fundamental_val > 0 else 0
        normalized_peaks.append((peak_frequencies[i], normalized, peak_values[i]))
    
    for ratio in harmonic_ratios:
        target_freq = fundamental_freq * ratio
        
        # Ищем ближайший пик к целевой гармонике
        closest_peak = None
        min_diff = float('inf')
        
        for freq, norm, val in normalized_peaks:
            if freq < 50:  # Пропускаем слишком низкие частоты
                continue
                
            freq_diff = abs(freq - target_freq)
            relative_diff = freq_diff / target_freq
            
            if relative_diff < 0.08 and freq_diff < min_diff:  # Допуск 8%
                min_diff = freq_diff
                closest_peak = (freq, norm, val)
        
        # Если нашли подходящий пик и его амплитуда достаточна
        if closest_peak and closest_peak[1] > 0.2:  # Амплитуда не менее 20% от основной
            harmonics_found += 1
    
    # 5. Критерии для дронов
    current_result = False
    
    # Критерии:
    # 1. Нашли хотя бы 3 гармоники
    # 2. Основная частота в разумном диапазоне (50-1500 Гц)
    # 3. Гармоники имеют достаточно равномерное распределение
    if (harmonics_found >= 3 and 
        50 < fundamental_freq < 1500 and 
        fundamental_val > 0.7):
        
        # Дополнительная проверка: гармоники должны быть примерно равномерными
        # Считаем отношение амплитуд соседних гармоник
        if harmonics_found >= 4:
            current_result = True
    
    # 6. Проверяем стабильность частоты
    if len(frequency_history) >= 5:
        recent_freqs = list(frequency_history)[-5:]
        freq_std = np.std(recent_freqs)
        if freq_std > 80:  # Если частота сильно скачет
            current_result = False
    
    # Добавляем текущий результат в историю
    harmonic_history.append(current_result)
    
    # 7. Временная фильтрация
    if len(harmonic_history) < HARMONIC_HISTORY_SIZE // 2:
        return False, fundamental_freq
    
    # Подсчитываем статистику
    positive_count = sum(harmonic_history)
    total_count = len(harmonic_history)
    positive_ratio = positive_count / total_count if total_count > 0 else 0
    
    # Решение на основе истории
    if positive_ratio >= CONFIRMATION_THRESHOLD:
        # Берем медианную частоту для стабильности
        if frequency_history:
            confirmed_frequency = np.median(list(frequency_history)[-10:])
        else:
            confirmed_frequency = fundamental_freq
        return True, confirmed_frequency
    else:
        return False, fundamental_freq

def safe_exit(): #Безопасно закрывает программу
    print("Закрываю программу...")
    try:
        if 'audio_stream' in globals():
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
    
    try:
        if 'window' in globals():
            window.quit()
            window.destroy()
            print("Окно закрыто")
    except Exception as e:
        print(f"Ошибка при закрытии окна: {e}")
    
    sys.exit(0)


# Обработка закрытия окна
window.protocol("WM_DELETE_WINDOW", safe_exit)

# Размеры интерфейса
WIDTH = 1600
HEIGHT = 1000
PLOT_WIDTH = int(WIDTH * 0.85)
PLOT_HEIGHT = int(HEIGHT * 0.85)

# Главный контейнер
main_container = tk.Frame(window, bg='#d0d0d0', bd=3, relief=tk.SUNKEN)
main_container.place(x=15, y=15, width=PLOT_WIDTH, height=PLOT_HEIGHT)


def create_stream(buffer, rate, device_index=None): #Инициализация аудиопотока с указанным микрофоном
    if device_index is None:
        device_index = current_device_index
    
    return audio_engine.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=buffer
    )


BUFFER_SIZE = BUFFER_SIZE
SAMPLE_RATE = SAMPLE_RATE
audio_stream = create_stream(BUFFER_SIZE, SAMPLE_RATE, current_device_index)

# Создание графиков
fig, (plot1, plot2) = plt.subplots(
    2, 1,
    facecolor='white',
    figsize=(10, 8),
    gridspec_kw={'height_ratios': [2, 1]}
)
fig.tight_layout(pad=3.0)

# Спектрограмма
spectrogram_buffer = np.zeros((FFT_SIZE // 2, 100))
spectrogram_buffer.fill(1e-3)
spectrogram_image = plot1.imshow(
    spectrogram_buffer,
    aspect='auto',
    cmap='seismic',
    origin='lower',
    extent=[0, 10, 0, SAMPLE_RATE // 2]
)

# Оформление спектрограммы
plot1.set_ylim(0, SAMPLE_RATE // 2)
plot1.set_xlim(0, 10)
plot1.set_facecolor('white')
plot1.set_title("Спектрограмма", color='black', fontsize=16, fontweight='bold')
plot1.set_xlabel("Время, с", color='black', fontsize=12)
plot1.set_ylabel("Частота, Гц", color='black', fontsize=12)

for spine in plot1.spines.values():
    spine.set_color('#404040')
plot1.xaxis.label.set_color('black')
plot1.yaxis.label.set_color('black')
plot1.tick_params(axis='x', colors='black')
plot1.tick_params(axis='y', colors='black')

# График амплитуды
line_amplitude, = plot2.plot([], [], color='#ff6600', lw=2)
plot2.set_xlim(0, SAMPLE_RATE // 2)
plot2.set_ylim(0, 10)
plot2.set_facecolor('white')
plot2.set_title("Частотный ответ", color='black', fontsize=16, fontweight='bold')
plot2.set_xlabel("Частота, Гц", color='black', fontsize=12)
plot2.set_ylabel("Амплитуда (лог.)", color='black', fontsize=12)
for spine in plot2.spines.values():
    spine.set_color('#404040')
plot2.xaxis.label.set_color('black')
plot2.yaxis.label.set_color('black')
plot2.tick_params(axis='x', colors='black')
plot2.tick_params(axis='y', colors='black')

# Привязка графиков к Tkinter
canvas = FigureCanvasTkAgg(fig, master=main_container)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Контролы справа
control_panel_x = PLOT_WIDTH + 35

# Ползунок буфера
slider_buffer = tk.Scale(
    window,
    from_=256,
    to=4096,
    orient=tk.HORIZONTAL,
    label='Buffer Size',
    length=280,
    bg='#2a2a2a',
    fg='#ffffff',
    troughcolor='#404040',
    font=("Courier", 9)
)
slider_buffer.set(BUFFER_SIZE)
slider_buffer.place(x=control_panel_x, y=20)

# Ползунок частоты дискретизации
slider_rate = tk.Scale(
    window,
    from_=8000,
    to=48000,
    orient=tk.HORIZONTAL,
    label='Sample Rate',
    length=280,
    bg='#2a2a2a',
    fg='#ffffff',
    troughcolor='#404040',
    font=("Courier", 9)
)
slider_rate.set(SAMPLE_RATE)
slider_rate.place(x=control_panel_x, y=100)

# Дополнительный индикатор для гармоник
harmonics_label = tk.Label(
    window,
    text="Проверка гармоник:",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 11, "bold")
)
harmonics_label.place(x=control_panel_x, y=180)

harmonics_indicator = tk.Canvas(
    window,
    width=20,
    height=20,
    bg='#d0d0d0',
    highlightthickness=0
)
harmonics_indicator.place(x=control_panel_x, y=210)
harmonics_light = harmonics_indicator.create_rectangle(2, 2, 18, 18, fill='#ff4444', outline='black', width=1)

harmonics_status = tk.Label(
    window,
    text=" - гармоники не проверяются",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
harmonics_status.place(x=control_panel_x + 25, y=210)

# Индикатор подтверждения гармоник
confirmation_label = tk.Label(
    window,
    text="Подтверждение:",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 11, "bold")
)
confirmation_label.place(x=control_panel_x, y=240)

confirmation_indicator = tk.Canvas(
    window,
    width=20,
    height=20,
    bg='#d0d0d0',
    highlightthickness=0
)
confirmation_indicator.place(x=control_panel_x, y=270)
confirmation_light = confirmation_indicator.create_rectangle(2, 2, 18, 18, fill='#ff4444', outline='black', width=1)

confirmation_status = tk.Label(
    window,
    text=" - не подтверждены",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
confirmation_status.place(x=control_panel_x + 25, y=270)

# Информация о частоте
frequency_label = tk.Label(
    window,
    text="Частота: --- Гц",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
frequency_label.place(x=control_panel_x, y=300)

# Прогресс подтверждения
confirmation_progress = tk.Canvas(
    window,
    width=100,
    height=15,
    bg='white',
    highlightthickness=1,
    highlightbackground='#404040'
)
confirmation_progress.place(x=control_panel_x, y=320)
confirmation_bar = confirmation_progress.create_rectangle(0, 0, 0, 15, fill='#ff4444')

# СЕКЦИЯ ИНДИКАТОРА
# Заголовок индикатора
indicator_label = tk.Label(
    window,
    text="Индикатор:",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 11, "bold")
)
indicator_label.place(x=control_panel_x, y=350)

# Зеленый квадратик
green_indicator = tk.Canvas(
    window,
    width=20,
    height=20,
    bg='#d0d0d0',
    highlightthickness=0
)
green_indicator.place(x=control_panel_x, y=380)
green_indicator.create_rectangle(2, 2, 18, 18, fill='#44ff44', outline='black', width=1)

green_label = tk.Label(
    window,
    text=" - есть дрон",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
green_label.place(x=control_panel_x + 25, y=380)

# Красный квадратик
red_indicator = tk.Canvas(
    window,
    width=20,
    height=20,
    bg='#d0d0d0',
    highlightthickness=0
)
red_indicator.place(x=control_panel_x, y=410)
red_indicator.create_rectangle(2, 2, 18, 18, fill='#ff4444', outline='black', width=1)

red_label = tk.Label(
    window,
    text=" - нет дрона",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
red_label.place(x=control_panel_x + 25, y=410)

# Сам индикатор статуса
status_indicator = tk.Canvas(
    window,
    width=110,
    height=60,
    bg='#2a2a2a',
    highlightthickness=1,
    highlightbackground='#404040'
)
status_indicator.place(x=control_panel_x, y=440)
status_light = status_indicator.create_oval(20, 15, 45, 45, fill='#ff4444', outline='#ffffff', width=2)

# СЕКЦИЯ ВЫБОРА МИКРОФОНА
# Заголовок выбора микрофона
mic_label = tk.Label(
    window,
    text="Выберите микрофон:",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 11, "bold")
)
mic_label.place(x=control_panel_x, y=520)

# Создаем список для выпадающего меню
if microphones:
    mic_names = [f"{i}: {name[:50]}" for i, name in microphones]
    mic_var = tk.StringVar(value=mic_names[0])  # Устанавливаем первый микрофон по умолчанию
else:
    mic_names = ["Микрофоны не найдены"]
    mic_var = tk.StringVar(value=mic_names[0])

mic_dropdown = ttk.Combobox(
    window,
    textvariable=mic_var,
    values=mic_names,
    state="readonly",
    width=35,
    font=("Arial", 9)
)
mic_dropdown.place(x=control_panel_x, y=550)

# Информация о текущем микрофоне
current_mic_label = tk.Label(
    window,
    text=f"Текущий: {mic_names[0][:30]}" if microphones else "Микрофоны не найдены",
    bg='#d0d0d0',
    fg='black',
    font=("Arial", 9)
)
current_mic_label.place(x=control_panel_x, y=580)

# Временные переменры
session_start = time.time()
elapsed_time = 0
detection_time = 0
detection_active = False

# Целевые полосы частот
frequency_bands = [
    (banda_1_start, banda_1_end, amplitud_1),
    (banda_2_start, banda_2_end, amplitud_2),
    (banda_3_start, banda_3_end, amplitud_3),
]

# Функция обновления параметров
def sync_parameters(val=None):
    global BUFFER_SIZE, SAMPLE_RATE, audio_stream, spectrogram_image, spectrogram_buffer, elapsed_time, current_device_index
    global harmonic_history, frequency_history, confirmed_frequency, bands_detected_history

    BUFFER_SIZE = int(slider_buffer.get())
    SAMPLE_RATE = int(slider_rate.get())
    
    # Обновляем выбор микрофона если он был изменен
    if microphones and mic_names[0] != "Микрофоны не найдены":
        selected_mic = mic_var.get()
        for mic_str in mic_names:
            if mic_str == selected_mic:
                # Извлекаем индекс из строки
                device_index = int(mic_str.split(":")[0])
                current_device_index = device_index
                # Обновляем информацию о текущем микрофоне
                current_mic_label.config(text=f"Текущий: {mic_str[4:34]}...")
                break
    
    audio_stream.stop_stream()
    audio_stream.close()
    audio_stream = create_stream(BUFFER_SIZE, SAMPLE_RATE, current_device_index)

    plot1.set_ylim(0, SAMPLE_RATE // 2)
    spectrogram_buffer = np.zeros((FFT_SIZE // 2, 100))
    spectrogram_buffer.fill(1e-3)
    spectrogram_image.set_data(spectrogram_buffer)
    spectrogram_image.set_clim(vmin=np.min(spectrogram_buffer), vmax=np.max(spectrogram_buffer))
    spectrogram_image.set_extent([0, 10, 0, SAMPLE_RATE // 2])

    # Сбрасываем истории при изменении параметров
    harmonic_history.clear()
    frequency_history.clear()
    confirmed_frequency = None
    bands_detected_history.clear()

    refresh_plots(np.zeros(BUFFER_SIZE))
    
    audio_stream.stop_stream()
    audio_stream.close()
    audio_stream = create_stream(BUFFER_SIZE, SAMPLE_RATE, current_device_index)

    plot1.set_ylim(0, SAMPLE_RATE // 2)
    spectrogram_buffer = np.zeros((FFT_SIZE // 2, 100))
    spectrogram_buffer.fill(1e-3)
    spectrogram_image.set_data(spectrogram_buffer)
    spectrogram_image.set_clim(vmin=np.min(spectrogram_buffer), vmax=np.max(spectrogram_buffer))
    spectrogram_image.set_extent([0, 10, 0, SAMPLE_RATE // 2])

    # Сбрасываем историю при изменении параметров
    harmonic_history.clear()
    frequency_history.clear()
    confirmed_frequency = None

    refresh_plots(np.zeros(BUFFER_SIZE))


slider_buffer.config(command=sync_parameters)
slider_rate.config(command=sync_parameters)
mic_dropdown.bind("<<ComboboxSelected>>", sync_parameters)

# Линии пороговых значений
thresholds = [
    {"freq_low": banda_1_start, "freq_high": banda_1_end, "level": amplitud_1, "color": "#ff6666", "style": "--", "name": "Threshold A"},
    {"freq_low": banda_2_start, "freq_high": banda_2_end, "level": amplitud_2, "color": "#66ff66", "style": "--", "name": "Threshold B"},
]

threshold_objects = []
for threshold in thresholds:
    line_obj, = plot2.plot(
        [threshold["freq_low"], threshold["freq_high"]],
        [threshold["level"], threshold["level"]],
        color=threshold["color"],
        linestyle=threshold["style"],
        label=threshold["name"],
        lw=1.5
    )
    threshold_objects.append(line_obj)


# Функция обновления графиков
def refresh_plots(audio_data):
    global SAMPLE_RATE, BUFFER_SIZE, spectrogram_buffer, elapsed_time, detection_active, detection_time
    global detection_start_time, bands_detected_history

    # FFT преобразование
    frequency_spectrum = fft(audio_data, n=FFT_SIZE)
    magnitude = 2.0 / BUFFER_SIZE * np.abs(frequency_spectrum[:FFT_SIZE // 2])
    magnitude_log = np.log1p(magnitude)

    # Обновление спектрограммы
    spectrogram_buffer = np.roll(spectrogram_buffer, -1, axis=1)
    spectrogram_buffer[:, -1] = magnitude_log

    spectrogram_image.set_clim(vmin=np.min(spectrogram_buffer), vmax=np.max(spectrogram_buffer))
    spectrogram_image.set_data(spectrogram_buffer)

    # Обновление временной шкалы
    elapsed_time += BUFFER_SIZE / SAMPLE_RATE
    spectrogram_image.set_extent([0, min(10, elapsed_time), 0, SAMPLE_RATE // 2])

    # Обновление графика амплитуды
    frequency_axis = np.linspace(0, SAMPLE_RATE // 2, FFT_SIZE // 2)
    line_amplitude.set_data(frequency_axis, magnitude_log)

    # Проверка условий обнаружения в полосах частот
    # С учетом средней амплитуды в полосе
    bands_detection = []
    
    for band_low, band_high, threshold in frequency_bands:
        # Находим индексы в полосе частот
        band_indices = np.where((frequency_axis >= band_low) & (frequency_axis <= band_high))
        
        if len(band_indices[0]) > 0:
            # Берем среднюю амплитуду в полосе
            band_amplitude = np.mean(magnitude_log[band_indices])
            # И максимальную амплитуду
            max_amplitude = np.max(magnitude_log[band_indices])
            
            # Условие: средняя амплитуда > порога ИЛИ есть явный пик
            band_detected = band_amplitude > (threshold * 0.7) or max_amplitude > (threshold * 1.3)
            bands_detection.append(band_detected)
        else:
            bands_detection.append(False)
    
    # Проверяем, все ли полосы обнаружены
    all_bands_detected = all(bands_detection)
    
    # Добавляем результат в историю для временной фильтрации
    # История хранит последние 30 проверок (примерно 1.5 секунды при 50мс интервале)
    if 'bands_detected_history' not in globals():
        bands_detected_history = deque(maxlen=30)  # Храним 30 последних проверок
    
    bands_detected_history.append(all_bands_detected)
    
    # ПРОВЕРКА ГАРМОНИК С ЗАДЕРЖКОЙ
    harmonics_confirmed, current_freq = check_harmonics_with_delay(frequency_axis, magnitude_log)
    
    # Обновление индикатора гармоник
    if HARMONICS_CHECK:
        if harmonic_history:
            current_result = harmonic_history[-1]
        
            if current_result:
                harmonics_indicator.itemconfig(harmonics_light, fill='#44ff44')
                harmonics_status.config(text=" - гармоники есть")
            else:
                harmonics_indicator.itemconfig(harmonics_light, fill='#ff4444')
                harmonics_status.config(text=" - гармоник нет")
        else:
            harmonics_indicator.itemconfig(harmonics_light, fill='#888888')
            harmonics_status.config(text=" - сбор данных...")
    
        # Обновление индикатора подтверждения
        if harmonics_confirmed:
            confirmation_indicator.itemconfig(confirmation_light, fill='#44ff44')
            confirmation_status.config(text=" - подтверждены")
            frequency_label.config(text=f"Частота: {current_freq:.1f} Гц")
        else:
            confirmation_indicator.itemconfig(confirmation_light, fill='#ff4444')
            confirmation_status.config(text=" - не подтверждены")
            if current_freq:
                frequency_label.config(text=f"Частота: {current_freq:.1f} Гц")
            else:
                frequency_label.config(text="Частота: --- Гц")

    # Управление индикатором 
    # Проверяем историю полос - сигнал должен быть стабильным
    if len(bands_detected_history) >= 30:  # Ждем пока накопится история (1.5 секунды)
        # Считаем, сколько раз полосы были обнаружены за последние 3 секунды (примерно 60 проверок)
        recent_detections = list(bands_detected_history)[-60:] if len(bands_detected_history) >= 60 else list(bands_detected_history)
        positive_ratio = sum(recent_detections) / len(recent_detections) if recent_detections else 0
        
        # Сигнал стабилен, если полосы обнаружены в 80% случаев за последние 3 секунды
        bands_stable = positive_ratio >= 0.8
        
        if bands_stable:
            if HARMONICS_CHECK:
                # Если включена проверка гармоник, требуем подтверждения гармоник
                if harmonics_confirmed:
                    # Все условия выполнены - включаем зеленый индикатор
                    status_indicator.itemconfig(status_light, fill='#44ff44')
                    detection_active = True
                else:
                    # Полосы есть, но гармоники не подтверждены
                    status_indicator.itemconfig(status_light, fill='#ff4444')
                    detection_active = False
            else:
                # Проверка гармоник отключена - достаточно стабильных полос
                status_indicator.itemconfig(status_light, fill='#44ff44')
                detection_active = True
        else:
            # Полосы нестабильны или отсутствуют
            status_indicator.itemconfig(status_light, fill='#ff4444')
            detection_active = False
    else:
        # Недостаточно истории для принятия решения
        status_indicator.itemconfig(status_light, fill='#ff4444')
        detection_active = False

    canvas.draw()
    canvas.flush_events()


# Основной цикл обработки
def process_audio():
    try:
        raw_audio = np.frombuffer(
            audio_stream.read(BUFFER_SIZE, exception_on_overflow=False),
            dtype=np.int16
        )
        refresh_plots(raw_audio)
        window.after(50, process_audio) 
    except Exception as e:
        print(f"Audio Error: {e}")
        # Попытка переподключения при ошибке
        window.after(1000, process_audio)


window.after(100, process_audio)
window.geometry(f'{WIDTH}x{HEIGHT}')

try:
    print("Программа запущена. Для выхода закрой окно или нажми Ctrl+C в консоли")
    print(f"Проверка гармоник: {'ВКЛЮЧЕНА' if HARMONICS_CHECK else 'ВЫКЛЮЧЕНА'}")
    print(f"Временная фильтрация: {HARMONIC_HISTORY_SIZE} кадров (~1 секунда)")
    print(f"Порог подтверждения: {CONFIRMATION_THRESHOLD*100}%")
    if microphones:
        print(f"Найдено микрофонов: {len(microphones)}")
        for idx, (device_id, name) in enumerate(microphones):
            print(f"  {device_id}: {name}")
    else:
        print("Микрофоны не найдены!")
    window.mainloop()
except KeyboardInterrupt:
    print("Программа прервана по Ctrl+C")
except Exception as e:
    print(f"Произошла ошибка: {e}")
finally:
    safe_exit()


