// ГЛОБАЛЬНЫЕ ФУНКЦИИ ДЛЯ ONCLICK
function clearLog() {
    if (window.droneSystem) {
        window.droneSystem.clearLog();
    }
}

class DroneDetectionSystem {
    constructor() {
        this.serverUrl = window.location.origin;
        this.installations = ['north', 'east', 'south', 'west'];
        this.updateInterval = 2000;
        this.systemUptimeSeconds = 0;
        this.lastLogHash = '';
        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.startAll());
        } else {
            this.startAll();
        }
    }

    startAll() {
        console.log('Система запущена');
        this.startUptimeTimer();
        this.startDataPolling();
        this.fetchData();
    }

    startUptimeTimer() {
        let seconds = 0;
        setInterval(() => {
            seconds++;
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;

            // ИЩЕМ ЭЛЕМЕНТ ПРЯМО ПЕРЕД СЛОВОМ "Время работы"
            const allSpans = document.querySelectorAll('span');
            for (const span of allSpans) {
                // Проверяем текущий и соседние элементы
                if (span.parentElement?.textContent?.includes('Время работы')) {
                    // Ищем спан с временем в этом блоке
                    const timeSpans = span.parentElement.querySelectorAll('span');
                    if (timeSpans.length > 0) {
                        // Обычно это последний спан в блоке
                        for (let i = timeSpans.length - 1; i >= 0; i--) {
                            if (timeSpans[i].textContent.match(/\d{2}:\d{2}:\d{2}/)) {
                                timeSpans[i].textContent = timeStr;
                                return;
                            }
                        }
                    }
                }
            }
        }, 1000);
    }

    async fetchData() {
        try {
            const response = await fetch(`${this.serverUrl}/api/status`);
            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();
            console.log('Data received:', data);

            this.updateUI(data);
            this.updateLog(data.log);
            this.updateStats(data);

        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    updateUI(data) {
        const installationsData = data.installations;

        this.installations.forEach(id => {
            const installation = installationsData[id];

            const dot = document.querySelector(`.${id} .status-dot`);
            const installDiv = document.querySelector(`.${id}`);

            if (!installDiv) return;

            const spans = installDiv.querySelectorAll('span');
            if (spans.length < 3) return;

            const timeSpan = spans[0];
            const freqSpan = spans[1];
            const confSpan = spans[2];

            const isDetected = installation.status === 'green';

            if (!dot) return;

            if (isDetected) {
                dot.className = 'status-dot';
                dot.style.backgroundColor = '#20d997';
                dot.style.animation = 'pulse-green 0.8s infinite';
                installDiv.classList.add('alarm');
            } else {
                dot.className = 'status-dot';
                dot.style.backgroundColor = '#ff4757';
                dot.style.animation = 'none';
                installDiv.classList.remove('alarm');
            }

            timeSpan.textContent = installation.detection_time || '--:--:--';
            freqSpan.textContent = installation.frequency ? `${installation.frequency.toFixed(1)} Гц` : '--- Гц';
            confSpan.textContent = installation.confidence ? `${Math.round(installation.confidence * 100)}%` : '---%';

            console.log(`${id}: ${installation.status}, time: ${installation.detection_time}`);
        });
    }

    updateLog(logEntries) {
        const logContent = document.getElementById('log-content');
        if (!logContent) return;

        if (!logEntries || logEntries.length === 0) {
            logContent.innerHTML = '<div style="text-align: center; opacity: 0.5;">Ожидание событий...</div>';
            return;
        }

        const logHash = JSON.stringify(logEntries);
        if (logHash === this.lastLogHash) {
            return;
        }

        this.lastLogHash = logHash;

        const html = logEntries.map(entry => {
            const icon = entry.message.includes('Обнаружен') ? '🚁' :
                entry.message.includes('покинул') ? '⚠️' : 'ℹ️';

            return `
                <div class="log-entry ${entry.message.includes('Обнаружен') ? 'detection' : 'clear'}">
                    <span class="log-time">${entry.timestamp}</span>
                    <span class="log-install">${entry.installation}</span>
                    <span class="log-icon">${icon}</span>
                    <span class="log-msg">${entry.message}</span>
                </div>
            `;
        }).join('');

        logContent.innerHTML = html;
        console.log('Log updated:', logEntries.length, 'entries');
    }

    updateStats(data) {
        const statCards = document.querySelectorAll('.stat-value');
        if (statCards.length >= 2) {
            statCards[0].textContent = data.total_detections || 0;
            statCards[1].textContent = data.active_now || 0;
            console.log('Stats updated:', data.total_detections, data.active_now);
        }
    }

    startDataPolling() {
        setInterval(() => this.fetchData(), this.updateInterval);
    }

    async clearLog() {
        try {
            console.log('Clearing log...');
            const response = await fetch(`${this.serverUrl}/api/clear-log`, { method: 'POST' });
            if (!response.ok) throw new Error('Failed to clear log');

            const logContent = document.getElementById('log-content');
            if (logContent) {
                logContent.innerHTML = '<div style="text-align: center; opacity: 0.5;">Ожидание событий...</div>';
            }
            this.lastLogHash = '';
            console.log('Log cleared successfully');
        } catch (error) {
            console.error('Error clearing log:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.droneSystem = new DroneDetectionSystem();
    console.log('DroneDetectionSystem initialized');
});
