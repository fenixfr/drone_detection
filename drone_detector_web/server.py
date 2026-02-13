#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request
from datetime import datetime
import json
import threading
import time
from queue import Queue, Empty
import logging

app = Flask(__name__, 
            template_folder='.',
            static_folder='.',
            static_url_path='')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

installations = {
    "north": {
        "status": "red", 
        "last_update": None, 
        "detection_time": None, 
        "frequency": 0, 
        "confidence": 0,
        "last_detection_state": None
    },
    "east": {
        "status": "red", 
        "last_update": None, 
        "detection_time": None, 
        "frequency": 0, 
        "confidence": 0,
        "last_detection_state": None
    },
    "south": {
        "status": "red", 
        "last_update": None, 
        "detection_time": None, 
        "frequency": 0, 
        "confidence": 0,
        "last_detection_state": None
    },
    "west": {
        "status": "red", 
        "last_update": None, 
        "detection_time": None, 
        "frequency": 0, 
        "confidence": 0,
        "last_detection_state": None
    }
}

detection_log = []
MAX_LOG_ENTRIES = 100
total_detections = 0

data_lock = threading.Lock()
update_queue = Queue(maxsize=500)

def add_log_entry(installation_id, message, frequency=None, confidence=None):
    global total_detections
    
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "installation": installation_id.upper(),
        "message": message,
        "frequency": frequency if frequency is not None else 0,
        "confidence": confidence if confidence is not None else 0
    }
    
    if "Обнаружен дрон!" in message:
        total_detections += 1
    
    detection_log.append(entry)
    
    if len(detection_log) > MAX_LOG_ENTRIES:
        detection_log.pop(0)

def process_updates():
    logger.info("✅ Фоновый поток обработки запущен")
    
    while True:
        try:
            data = update_queue.get(timeout=1)
            
            installation_id = data.get('installation_id', '').lower()
            detected = data.get('detected', False)
            frequency = float(data.get('frequency', 0))
            confidence = float(data.get('confidence', 0))
            
            if installation_id not in installations:
                logger.warning(f"⚠️ Неизвестная установка: {installation_id}")
                continue
            
            with data_lock:
                current_time = datetime.now()
                prev_detected = installations[installation_id]["last_detection_state"]
                
                if prev_detected == detected:
                    installations[installation_id]["last_update"] = current_time.strftime("%H:%M:%S")
                    if detected:
                        installations[installation_id]["frequency"] = frequency
                        installations[installation_id]["confidence"] = confidence
                    continue
                
                installations[installation_id]["last_detection_state"] = detected
                
                if detected:
                    installations[installation_id]["status"] = "green"
                    installations[installation_id]["detection_time"] = current_time.strftime("%H:%M:%S")
                    installations[installation_id]["frequency"] = frequency
                    installations[installation_id]["confidence"] = confidence
                    message = f"🚁 Обнаружен дрон! Частота: {frequency:.1f} Гц, Уверенность: {confidence:.0%}"
                    add_log_entry(installation_id, message, frequency, confidence)
                    logger.info(f"🟢 [{installation_id.upper()}] DETECTED - Частота: {frequency:.1f} Гц, Уверенность: {confidence:.0%}")
                    
                else:
                    installations[installation_id]["status"] = "red"
                    installations[installation_id]["frequency"] = 0
                    installations[installation_id]["confidence"] = 0
                    
                    if installations[installation_id]["detection_time"]:
                        message = "⚠️ Дрон покинул зону обнаружения"
                        add_log_entry(installation_id, message)
                        installations[installation_id]["detection_time"] = None
                        logger.info(f"🔴 [{installation_id.upper()}] CLEAR - Дрон ушёл")
                
                installations[installation_id]["last_update"] = current_time.strftime("%H:%M:%S")
        
        except Empty:
            pass
        except Exception as e:
            logger.error(f"❌ Ошибка: {type(e).__name__}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    with data_lock:
        response_data = {
            "installations": dict(installations),
            "log": list(detection_log[-10:]),
            "total_detections": total_detections,
            "active_now": sum(1 for inst in installations.values() if inst["status"] == "green"),
            "server_time": datetime.now().strftime("%H:%M:%S")
        }
        return jsonify(response_data)

@app.route('/api/update', methods=['POST'])
def update_status():
    try:
        data = request.json
        
        if not data or 'installation_id' not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        installation_id = data.get('installation_id', '').lower()
        
        if installation_id not in installations:
            return jsonify({"error": "Invalid installation ID"}), 400
        
        if not update_queue.full():
            update_queue.put_nowait(data)
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": True, "warning": "Queue full"}), 200
    
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history')
def get_history():
    with data_lock:
        return jsonify({
            "log": list(detection_log),
            "total_detections": total_detections
        })

@app.route('/api/clear-log', methods=['POST'])
def clear_log():
    global total_detections
    with data_lock:
        detection_log.clear()
        total_detections = 0
        logger.info("🗑️ Лог очищен")
        return jsonify({"success": True})

if __name__ == '__main__':
    update_thread = threading.Thread(target=process_updates, daemon=True)
    update_thread.start()
    
    print("\n" + "=" * 60)
    print("СЕРВЕР ДЕТЕКТОРА ДРОНОВ ЗАПУЩЕН")
    print("=" * 60)
    print("Доступные эндпоинты:")
    print(" • http://localhost:5000/ - Веб-интерфейс")
    print(" • http://localhost:5000/api/status - Статус установок (GET)")
    print(" • http://localhost:5000/api/update - Обновление статуса (POST)")
    print(" • http://localhost:5000/api/history - История (GET)")
    print(" • http://localhost:5000/api/clear-log - Очистка логов (POST)")
    print("=" * 60)
    print("⏳ Ожидание данных от детектора...")
    print("=" * 60 + "\n")
    
    from waitress import serve
    
    try:
        serve(app, host='0.0.0.0', port=5000, threads=20, _quiet=True)
    except KeyboardInterrupt:
        print("\n\nСервер остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
