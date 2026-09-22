let tracksOnlyFilter = false;

document.addEventListener("DOMContentLoaded", () => {
    const urlParams = new URLSearchParams(window.location.search);
    const urlDate = urlParams.get('date');

    const datePicker = document.getElementById("datePicker");

    if (datePicker) {
        if (urlDate) {
            datePicker.value = urlDate;
        } else {
            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            datePicker.value = `${year}-${month}-${day}`;
        }
    }

    const isGitHub = urlParams.get('env') === 'github' || window.location.hostname.includes('github.io');
    const badge = document.getElementById('env-badge');
    const classificationTitle = document.getElementById('classification-title')

    if (isGitHub) {
        if (badge) { badge.innerText = "🛰️ GitHub Static Mode"; badge.style.backgroundColor = "#15803d"; }
        if (classificationTitle) {
            classificationTitle.textContent = "📐 Morphological Structural Classification (Today)";
        }
        runServerlessGitHubPipeline();
    } else {
        if (badge) { badge.innerText = "⚡ Local Mode"; badge.style.backgroundColor = "#2563eb"; }
        if (classificationTitle) {
            classificationTitle.textContent = "📐 Morphological Structural Classification (All time)";
        }
        runActiveBackendPipeline();
    }

    const savedScroll = sessionStorage.getItem("scrollPos");
    if (savedScroll) {
        window.scrollTo(0, parseInt(savedScroll, 10));
        sessionStorage.removeItem("scrollPos");
    }
});

window.toggleTrackFilter = function () {
    tracksOnlyFilter = !tracksOnlyFilter;

    const btn = document.getElementById("filterTracksBtn");
    const label = document.getElementById("filterLabel");

    if (btn && label) {
        if (tracksOnlyFilter) {
            btn.classList.add("active");
            label.innerText = "Showing Tracks Only";
        } else {
            btn.classList.remove("active");
            label.innerText = "Show Tracks Only";
        }
    }

    const urlParams = new URLSearchParams(window.location.search);
    const isGitHub = urlParams.get('env') === 'github' || window.location.hostname.includes('github.io');

    if (isGitHub) {
        runServerlessGitHubPipeline();
    } else {
        runActiveBackendPipeline();
    }
};

function playParticleChime() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const audioCtx = new AudioContext();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.4);
    } catch (e) {
        console.error("Audio core initialize failure:", e);
    }
}

function checkAudioAlerts(newCount) {
    const baselineHitsCount = parseInt(localStorage.getItem("muon_baseline_count"), 10);
    if (!isNaN(baselineHitsCount) && newCount > baselineHitsCount) {
        playParticleChime();
    }
    localStorage.setItem("muon_baseline_count", newCount);
}

function loadSelectedDate() {
    const dateVal = document.getElementById("datePicker").value;
    if (dateVal) {
        sessionStorage.setItem("scrollPos", window.scrollY);
        window.location.href = "/?date=" + dateVal;
    }
}
window.loadSelectedDate = loadSelectedDate;

window.openZoom = function (src, name, prob) {
    document.getElementById("modalImg").src = src;
    const probDisplay = prob ? ` | PROBABILITY: ${prob}` : '';
    document.getElementById("modalCaption").innerText = `SENSOR CROP CAPTURE: ${name}${probDisplay}`;
    document.getElementById("zoomModal").style.display = "flex";
};

window.closeZoom = function () {
    document.getElementById("zoomModal").style.display = "none";
};

let poissonChartInstance = null;
let hourlyChartInstance = null;
let intervalChartInstance = null;

function renderPoissonChart(labels, observed, expected) {
    const canvas = document.getElementById("poissonChart");
    if (!canvas) return;

    if (poissonChartInstance) {
        poissonChartInstance.data.labels = labels;
        poissonChartInstance.data.datasets[0].data = observed;
        poissonChartInstance.data.datasets[1].data = expected;
        poissonChartInstance.update();
        return;
    }

    const ctx = canvas.getContext("2d");
    const fillGradient = ctx.createLinearGradient(0, 0, 0, 260);
    fillGradient.addColorStop(0, "rgba(124, 58, 237, 0.25)");
    fillGradient.addColorStop(1, "rgba(124, 58, 237, 0.0)");

    poissonChartInstance = new Chart(ctx, {
        data: {
            labels: labels,
            datasets: [
                {
                    type: "bar",
                    label: "Observed Flux",
                    data: observed,
                    backgroundColor: "rgba(37, 99, 235, 0.75)",
                    borderColor: "#2563eb",
                    borderWidth: 1.5,
                    barPercentage: 0.6,
                    order: 2
                },
                {
                    type: "line",
                    label: "Theoretical Poisson Fit",
                    data: expected,
                    borderColor: "#7c3aed",
                    backgroundColor: fillGradient,
                    fill: true,
                    borderWidth: 1.8,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: "#7c3aed",
                    tension: 0.25,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "nearest", intersect: false },
            scales: {
                x: {
                    grid: { color: "#f1f5f9", drawTicks: true, tickLength: 6 },
                    ticks: { color: "#64748b", font: { size: 11 } },
                    title: { display: true, text: "Detections Registered per Hour", color: "#334155", font: { size: 12, weight: "600" } }
                },
                y: {
                    type: "linear",
                    position: "left",
                    grid: { color: "#f1f5f9" },
                    ticks: { color: "#64748b", font: { size: 11 }, callback: (val) => val + "%" },
                    title: { display: true, text: "Frequency (%)", color: "#334155", font: { size: 12, weight: "600" } }
                }
            },
            plugins: {
                legend: { display: true, position: "top", align: "end", labels: { color: "#334155", font: { size: 11 }, boxWidth: 12, usePointStyle: true } },
                tooltip: { backgroundColor: "#0f172a", titleColor: "#f8fafc", bodyColor: "#f8fafc", padding: 10, cornerRadius: 6, callbacks: { label: (context) => ` ${context.dataset.label}: ${context.parsed.y}%` } }
            }
        }
    });
}

function renderHourlyChart(labels, counts) {
    const canvas = document.getElementById("hourlyChart");
    if (!canvas) return;

    if (hourlyChartInstance) {
        hourlyChartInstance.data.labels = labels;
        hourlyChartInstance.data.datasets[0].data = counts;
        hourlyChartInstance.update();
        return;
    }

    const ctx = canvas.getContext("2d");

    hourlyChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Hourly Flux Distribution",
                data: counts,
                backgroundColor: "rgba(13, 148, 136, 0.75)",
                borderColor: "#0d9488",
                borderWidth: 1.5,
                barPercentage: 0.7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "nearest", intersect: false },
            scales: {
                x: {
                    grid: { color: "#f1f5f9", drawTicks: true },
                    ticks: { color: "#64748b", font: { size: 10 }, maxTicksLimit: 12 },
                    title: { display: true, text: "Time of Day (UTC Profile)", color: "#334155", font: { size: 12, weight: "600" } }
                },
                y: {
                    grid: { color: "#f1f5f9" },
                    ticks: { color: "#64748b", font: { size: 11 } },
                    title: { display: true, text: "Events Registered", color: "#334155", font: { size: 12, weight: "600" } }
                }
            },
            plugins: {
                legend: { display: true, position: "top", align: "end", labels: { color: "#334155", font: { size: 11 }, boxWidth: 12, usePointStyle: true } },
                tooltip: { backgroundColor: "#0f172a", titleColor: "#f8fafc", bodyColor: "#f8fafc", padding: 10, cornerRadius: 6 }
            }
        }
    });
}

function renderIntervalChart(labels, observed, expected) {
    const canvas = document.getElementById("intervalChart");
    if (!canvas) return;

    if (intervalChartInstance) {
        intervalChartInstance.data.labels = labels;
        intervalChartInstance.data.datasets[0].data = observed;
        intervalChartInstance.data.datasets[1].data = expected;
        intervalChartInstance.update();
        return;
    }

    const ctx = canvas.getContext("2d");
    const fillGradient = ctx.createLinearGradient(0, 0, 400, 0);
    fillGradient.addColorStop(0, "rgba(217, 119, 6, 0.25)");
    fillGradient.addColorStop(1, "rgba(217, 119, 6, 0.0)");

    intervalChartInstance = new Chart(ctx, {
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'Observed Spacing Spans',
                    data: observed,
                    backgroundColor: '#3b82f6',
                    borderColor: '#2563eb',
                    borderWidth: 1.5,
                    barPercentage: 0.6,
                    order: 2
                },
                {
                    type: 'line',
                    label: 'Expected Quantum Decay Fit',
                    data: expected,
                    borderColor: '#d97706',
                    backgroundColor: fillGradient,
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#d97706',
                    tension: 0.3,
                    order: 1
                }
            ]
        },
        options: {
            indexAxis: 'x',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "#f1f5f9" },
                    ticks: { color: "#64748b", font: { size: 11 } },
                    title: { display: true, text: "Inter-Arrival Timing Delays", color: "#334155", font: { size: 12, weight: "600" } }
                },
                y: {
                    min: 0,
                    grid: { color: "#f1f5f9" },
                    ticks: { color: "#64748b", font: { size: 11 }, callback: (val) => val + "%" },
                    title: { display: true, text: "Distribution Weight (%)", color: "#334155", font: { size: 12, weight: "600" } }
                }
            },
            plugins: {
                legend: { display: true, position: "top", align: "end", labels: { color: "#334155", font: { size: 11 }, boxWidth: 12, usePointStyle: true } },
                tooltip: { backgroundColor: "#0f172a", titleColor: "#f8fafc", bodyColor: "#f8fafc", padding: 10, cornerRadius: 6, callbacks: { label: (context) => ` ${context.dataset.label}: ${context.parsed.y}%` } }
            }
        }
    });
}

function runActiveBackendPipeline() {
    const urlParams = new URLSearchParams(window.location.search);
    const activeDate = urlParams.get('date') || '';

    let queryParams = [];
    if (activeDate) queryParams.push(`date=${activeDate}`);
    if (tracksOnlyFilter) queryParams.push('filter=tracks');

    const queryString = queryParams.length > 0 ? `?${queryParams.join('&')}` : '';

    const exportBtn = document.getElementById('export-link');
    if (exportBtn) exportBtn.href = `/log${activeDate ? `?date=${activeDate}` : ''}`;

    fetch('/api/stats_endpoint_mock_or_real' + (activeDate ? `?date=${activeDate}` : ''))
        .then(res => res.json())
        .then(data => {
            document.getElementById('status').innerText = data.status;
            document.getElementById('total_muons').innerText = data.total_muons;
            document.getElementById('flux_rate').innerText = data.flux_rate;
            document.getElementById('mean_saturation').innerText = data.mean_saturation;
            document.getElementById('max_bright').innerText = data.all_time_max_bright;
            document.getElementById('max_size').innerText = data.all_time_max_size;

            ['artefacts', 'dots', 'tracks', 'worms'].forEach(cat => {
                let pctVal = data[`class_${cat}`] || "0%";
                let elem = document.getElementById(`class_${cat}`);
                if (elem) elem.innerText = pctVal;

                let bar = document.getElementById(`bar_${cat}`);
                if (bar) bar.style.width = pctVal;
            });
        }).catch(err => {
            document.getElementById('status').innerText = `Data Stream Empty for Target Date`;
            document.getElementById('today_muons').innerText = "0";
            document.getElementById('log_rows').innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b;">No log rows recorded for this observation date layer.</td></tr>`;
            document.getElementById('gallery_cards').innerHTML = `<p style="grid-column: 1/-1; color: #64748b; text-align:center;">No track snapshot frames captured.</p>`;
            document.getElementById('trngValue').innerText = "--";

            if (poissonChartInstance) poissonChartInstance.destroy();
            if (hourlyChartInstance) hourlyChartInstance.destroy();
            if (intervalChartInstance) intervalChartInstance.destroy();
        });

    fetch('/api/table_rows_mock_or_real' + queryString)
        .then(res => res.json())
        .then(data => {
            document.getElementById('log_rows').innerHTML = data.rows_html;
            document.getElementById('gallery_cards').innerHTML = data.cards_html;
            if (data.today_breakdown) document.getElementById('today_muons').innerText = data.today_breakdown;

            const cleanCount = parseInt(data.today_breakdown.split(' ')[0], 10) || 0;
            checkAudioAlerts(cleanCount);
        });

    fetch('/api/random')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ready') {
                document.getElementById('trngValue').innerText = `${data.random_1_100} (${data.random_byte})`;
                const type = data.source_event_type;
                const capType = type.charAt(0).toUpperCase() + type.slice(1);

                document.getElementById('trng_meta').innerText = `Source: ${capType}`;
                document.getElementById('pool_bits').innerText = `Entropy pool bits: ${data.entropy_pool_bits}`;
            }
        });

    fetch('/api/poisson')
        .then(res => res.json())
        .then(d => renderPoissonChart(d.labels, d.observed_bins, d.poisson_fit));

    fetch('/api/hourly' + (activeDate ? `?date=${activeDate}` : ''))
        .then(res => res.json())
        .then(d => renderHourlyChart(d.labels, d.counts));

    fetch('/api/intervals')
        .then(res => res.json())
        .then(d => renderIntervalChart(d.labels, d.bins, d.expected_fit));
}

function runServerlessGitHubPipeline() {
    const datePicker = document.getElementById("datePicker");
    const urlParams = new URLSearchParams(window.location.search);
    const targetDate = urlParams.get('date') || datePicker.value;
    datePicker.value = targetDate;

    const dateSegments = targetDate.split('-');
    const dailyCsvPath = `data/${dateSegments[0]}/${dateSegments[1]}/${dateSegments[2]}/muon_log.csv`;
    const historicCsvPath = `/static/historical_data.csv`;
    const imageRoot = `data/${dateSegments[0]}/${dateSegments[1]}/${dateSegments[2]}/`;

    document.getElementById('status').innerText = `GitHub Static Data Parsing`;

    const exportBtn = document.getElementById('export-link');
    if (exportBtn) exportBtn.href = dailyCsvPath;

    fetch(dailyCsvPath)
        .then(res => { if (!res.ok) throw new Error("No radiation log entries captured for this date framework target."); return res.text(); })
        .then(csvText => {
            const lines = csvText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            if (lines.length <= 1) return;

            const rows = [];
            let totalBrightness = 0, maxSize = 0, maxBright = 0;
            let cCounts = { artefacts: 0, dots: 0, tracks: 0, worms: 0 };
            let dailyTimestamps = [];
            // let hourlyBuckets = new Array(24).fill(0);

            for (let i = 1; i < lines.length; i++) {
                const cols = lines[i].split(',');
                if (cols.length < 4) continue;
                rows.push(cols);

                const size = parseInt(cols[1]) || 0;
                const bright = parseInt(cols[2]) || 0;
                const type = cols[6] ? cols[6].toLowerCase().trim() : "dots";

                totalBrightness += bright;
                if (size > maxSize) maxSize = size;
                if (bright > maxBright) maxBright = bright;
                if (cCounts.hasOwnProperty(type)) cCounts[type]++;

                const cleanTimeStr = cols[0].split('.')[0];
                const dt = new Date(cleanTimeStr.replace(/-/g, '/'));
                const ts = dt.getTime() / 1000;

                if (!isNaN(ts)) {
                    dailyTimestamps.push(ts);
                    // const hour = dt.getHours();
                    // if (hour >= 0 && hour < 24) {
                    //     hourlyBuckets[hour]++;
                    // }
                }
            }

            const totalEvents = rows.length;

            let breakdownParts = [];
            if (cCounts.tracks > 0) breakdownParts.push(`${cCounts.tracks} Track${cCounts.tracks !== 1 ? 's' : ''}`);
            if (cCounts.dots > 0) breakdownParts.push(`${cCounts.dots} Dot${cCounts.dots !== 1 ? 's' : ''}`);
            if (cCounts.worms > 0) breakdownParts.push(`${cCounts.worms} Worm${cCounts.worms !== 1 ? 's' : ''}`);
            if (cCounts.artefacts > 0) breakdownParts.push(`${cCounts.artefacts} Noise`);

            let githubSummaryText = totalEvents;
            if (breakdownParts.length > 0) {
                githubSummaryText += ` (${breakdownParts.join(', ')})`;
            }

            document.getElementById('today_muons').innerText = githubSummaryText;
            document.getElementById('max_bright').innerText = `${maxBright}`;
            document.getElementById('max_size').innerText = `${maxSize}px`;
            document.getElementById('mean_saturation').innerText = `${(totalBrightness / totalEvents).toFixed(1)}/255`;

            Object.keys(cCounts).forEach(k => {
                const pctStr = totalEvents > 0 ? ((cCounts[k] / totalEvents) * 100).toFixed(1) + "%" : "0%";
                const elem = document.getElementById(`class_${k}`);
                if (elem) elem.innerText = pctStr;

                const bar = document.getElementById(`bar_${k}`);
                if (bar) bar.style.width = pctStr;
            });

            if (dailyTimestamps.length > 0) {
                dailyTimestamps.sort((a, b) => a - b);

                const firstTs = dailyTimestamps[0];
                const lastTs = dailyTimestamps[dailyTimestamps.length - 1];
                const durationSeconds = Math.max(1, lastTs - firstTs);
                const durationHours = durationSeconds / 3600.0;

                const durationElem = document.getElementById('duration');
                if (durationElem) {
                    const hours = Math.floor(durationHours);
                    const mins = Math.floor((durationSeconds % 3600) / 60);
                    durationElem.innerText = `${hours}h ${mins}m`;
                }

                const fluxRate = (totalEvents / Math.max(1.0, durationHours)).toFixed(2);
                document.getElementById('flux_rate').innerText = `${fluxRate}/hr`;
            }

            // Render Log Rows & Gallery Cards
            let logRowsHtml = "", cardsHtml = "";

            const displayRows = rows.slice().reverse().filter(r => {
                if (!tracksOnlyFilter) return true;
                const rawType = r[6] ? r[6].toLowerCase().trim() : "dots";
                return rawType === 'tracks' || rawType === 'track';
            });

            displayRows.forEach(r => {
                const typeTitle = r[6] ? r[6].charAt(0).toUpperCase() + r[6].slice(1) : "Dots";
                const displayTime = r[0].split('.')[0];
                const probStr = r[7] ? parseFloat(r[7]).toFixed(2) + "%" : "100.00%";

                logRowsHtml += `<tr><td>${displayTime}</td><td>${r[1]} px</td><td>${r[2]}</td><td>${typeTitle}</td><td><strong>${probStr}</strong></td></tr>`;

                cardsHtml += `<div class="card" onclick="openZoom('${imageRoot}${r[3]}', '${r[3]}', '${probStr}')"><img src="${imageRoot}${r[3]}" onerror="this.src='static/fallback.png'"><span>${r[3]}</span></div>`;
            });

            if (!cardsHtml) cardsHtml = "<p style='grid-column: 1/-1; color: #888; text-align: center;'>No track images available.</p>";
            if (!logRowsHtml) logRowsHtml = "<tr><td colspan='5' style='color:#888; text-align:center;'>No track records found.</td></tr>";

            document.getElementById('log_rows').innerHTML = logRowsHtml;
            document.getElementById('gallery_cards').innerHTML = cardsHtml;

            // Render Hourly Chart using daily buckets
            // const hourlyLabels = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);
            // renderHourlyChart(hourlyLabels, hourlyBuckets);
        })
        .catch(err => {
            document.getElementById('status').innerText = err.message;
        });


    fetch(historicCsvPath)
        .then(res => { if (!res.ok) throw new Error("Historic datasets missing."); return res.text(); })
        .then(csvText => {
            const lines = csvText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            if (lines.length <= 1) return;

            let historicTimestamps = [];
            let historicRows = [];
            let hourlyMap = {};
            let todaysHourlyBuckets = new Array(24).fill(0); // Specific for today's hourly chart

            for (let i = 1; i < lines.length; i++) {
                const cols = lines[i].split(',');
                if (cols.length < 2) continue;
                historicRows.push(cols);

                const cleanTimeStr = cols[0].split('.')[0];
                const dt = new Date(cleanTimeStr.replace(/-/g, '/'));
                const ts = dt.getTime() / 1000;

                if (!isNaN(ts)) {
                    // Used for All-Time Poisson & Intervals
                    historicTimestamps.push(ts);
                    const hourKey = `${dt.getFullYear()}-${dt.getMonth()}-${dt.getDate()}-${dt.getHours()}`;
                    hourlyMap[hourKey] = (hourlyMap[hourKey] || 0) + 1;

                    // Extracted separately for Today's Hourly Chart + Track Filter
                    if (cleanTimeStr.startsWith(targetDate)) {
                        const rawType = cols[6] ? cols[6].toLowerCase().trim() : "dots";
                        const matchesFilter = !tracksOnlyFilter || (rawType === 'tracks' || rawType === 'track');

                        if (matchesFilter) {
                            const hour = dt.getHours();
                            if (hour >= 0 && hour < 24) {
                                todaysHourlyBuckets[hour]++;
                            }
                        }
                    }
                }
            }

            // Render Hourly Chart using TODAY'S filtered data only
            const hourlyLabels = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);
            renderHourlyChart(hourlyLabels, todaysHourlyBuckets);

            // TRNG Generation (latest entry)
            if (historicRows.length > 0) {
                const latestRow = historicRows[historicRows.length - 1];
                const timeParts = latestRow[0].split('.');
                if (timeParts.length === 2) {
                    const micros = parseInt(timeParts[1]) || 0;
                    const rand100 = (micros % 100) + 1;
                    const randByteHex = `0x${(micros % 256).toString(16).padStart(2, '0')}`;
                    const rawType = latestRow[6] ? latestRow[6].trim() : "unknown";
                    const capType = rawType.charAt(0).toUpperCase() + rawType.slice(1);

                    document.getElementById('trngValue').innerText = `${rand100} (${randByteHex})`;
                    document.getElementById('trng_meta').innerText = `Source: ${capType}`;
                    document.getElementById('pool_bits').innerText = `Entropy pool bits: ${historicRows.length * 8}`;
                }
            }

            // Render Poisson Fit Chart using ALL-TIME historic data
            const hourlyCountsArray = Object.values(hourlyMap);
            generateLocalPoissonCalculations(hourlyCountsArray);

            // Render Inter-Arrival Delay Chart using ALL-TIME timestamps
            historicTimestamps.sort((a, b) => a - b);
            generateLocalIntervalCalculations(historicTimestamps);
        })
        .catch(err => {
            console.warn("Could not compute metrics from historic_data.csv:", err);
        });
}

function generateLocalPoissonCalculations(hourlyCounts) {
    if (!hourlyCounts || hourlyCounts.length === 0) return;

    const totalHours = hourlyCounts.length;
    const totalEvents = hourlyCounts.reduce((a, b) => a + b, 0);
    const lambda = totalEvents / totalHours;

    const maxHits = Math.max(...hourlyCounts, 5);
    const bins = new Array(maxHits + 1).fill(0);

    hourlyCounts.forEach(count => {
        if (count <= maxHits) {
            bins[count]++;
        }
    });

    const observedPcts = bins.map(b => parseFloat(((b / totalHours) * 100).toFixed(1)));
    const expectedPcts = [];

    const factorial = (n) => (n <= 1 ? 1 : n * factorial(n - 1));

    for (let k = 0; k <= maxHits; k++) {
        const prob = (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
        expectedPcts.push(parseFloat((prob * 100).toFixed(1)));
    }

    const labels = Array.from({ length: maxHits + 1 }, (_, i) => `${i}`);
    renderPoissonChart(labels, observedPcts, expectedPcts);
}

function generateLocalIntervalCalculations(timestamps) {
    const labels = ["0-10m", "10-20m", "20-30m", "30-40m", "40-50m", "50-60m"];
    const bins = [0, 0, 0, 0, 0, 0];

    if (!timestamps || timestamps.length < 2) {
        renderIntervalChart(labels, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]);
        return;
    }

    for (let i = 1; i < timestamps.length; i++) {
        const deltaMin = (timestamps[i] - timestamps[i - 1]) / 60.0;
        if (deltaMin <= 10) bins[0]++;
        else if (deltaMin <= 20) bins[1]++;
        else if (deltaMin <= 30) bins[2]++;
        else if (deltaMin <= 40) bins[3]++;
        else if (deltaMin <= 50) bins[4]++;
        else if (deltaMin <= 60) bins[5]++;
    }

    const totalIntervals = bins.reduce((a, b) => a + b, 0);
    const observedPcts = bins.map(b => totalIntervals > 0 ? parseFloat(((b / totalIntervals) * 100).toFixed(1)) : 0);

    const timeSpanMin = (timestamps[timestamps.length - 1] - timestamps[0]) / 60.0;
    const expectedPcts = [];

    if (timeSpanMin > 0 && totalIntervals > 0) {
        const lam = (timestamps.length - 1) / timeSpanMin;
        const binWidth = 10;

        for (let i = 0; i < 6; i++) {
            const t1 = i * binWidth;
            const t2 = (i + 1) * binWidth;
            const prob = Math.exp(-lam * t1) - Math.exp(-lam * t2);
            expectedPcts.push(parseFloat((prob * 100).toFixed(1)));
        }
    } else {
        expectedPcts.push(0, 0, 0, 0, 0, 0);
    }

    renderIntervalChart(labels, observedPcts, expectedPcts);
}