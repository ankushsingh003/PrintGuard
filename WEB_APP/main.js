const API_URL = 'http://localhost:8000';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPanel = document.getElementById('upload-panel');
const resultPanel = document.getElementById('result-panel');
const verdictText = document.getElementById('verdict-text');
const verdictBadge = document.getElementById('verdict-badge');
const confidenceVal = document.getElementById('confidence-val');
const confidenceBar = document.getElementById('confidence-bar');
const scoresList = document.getElementById('scores-list');
const docPreview = document.getElementById('doc-preview');
const resetBtn = document.getElementById('reset-btn');
const loader = document.getElementById('loader');
const reportTimestamp = document.getElementById('report-timestamp');

// Quality colors for the breakdown bars
const QUALITY_COLORS = {
    'Excellent': '#3fb950',
    'Good': '#58a6ff',
    'Fair': '#f0883e',
    'Poor': '#f85149'
};

// --- Event Listeners ---

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) processFile(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) processFile(fileInput.files[0]);
});

resetBtn.addEventListener('click', () => {
    resultPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    fileInput.value = '';
    scoresList.innerHTML = '';
});

// --- Core Logic ---

async function processFile(file) {
    // Show loader
    loader.classList.remove('hidden');

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        docPreview.innerHTML = `<img src="${e.target.result}" alt="Document preview" />`;
    };
    reader.readAsDataURL(file);

    // Call API
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await response.json();
        updateUI(data);

        // Switch panels
        uploadPanel.classList.add('hidden');
        resultPanel.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        loader.classList.add('hidden');
    }
}

function updateUI(data) {
    // Main verdict
    verdictText.textContent = data.prediction;
    verdictText.style.color = QUALITY_COLORS[data.prediction] || '#e6edf3';

    // Badge styling
    verdictBadge.textContent = data.prediction.toUpperCase() + ' QUALITY';
    verdictBadge.className = `verdict-badge ${data.prediction.toLowerCase()}`;

    // Confidence
    const pct = (data.confidence * 100).toFixed(1);
    confidenceVal.textContent = `${pct}%`;
    confidenceBar.style.width = `${pct}%`;

    // Timestamp
    const now = new Date();
    reportTimestamp.textContent = `Report generated: ${now.toLocaleString()}`;

    // Score breakdown
    scoresList.innerHTML = '';
    const allScores = data.all_scores || {};

    const sorted = Object.entries(allScores).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([cls, score]) => {
        const pct = (score * 100).toFixed(1);
        const color = QUALITY_COLORS[cls] || '#58a6ff';
        const item = document.createElement('div');
        item.className = 'score-item';
        item.innerHTML = `
      <div class="score-info">
        <span class="score-name">${cls}</span>
        <span class="score-val">${pct}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${pct}%; background: ${color};"></div>
      </div>
    `;
        scoresList.appendChild(item);
    });
}
