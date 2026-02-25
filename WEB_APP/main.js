const API_URL = 'http://localhost:8000';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const uploadCard = document.querySelector('.upload-card');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('reset-btn');

const topPrediction = document.getElementById('top-prediction');
const topConfidence = document.getElementById('top-confidence');
const previewImage = document.getElementById('preview-image');
const scoresList = document.getElementById('scores-list');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

resetBtn.addEventListener('click', resetUI);

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file.');
    return;
  }

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.style.backgroundImage = `url(${e.target.result})`;
  };
  reader.readAsDataURL(file);

  // Send to API
  await predict(file);
}

async function predict(file) {
  loader.classList.remove('hidden');
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Prediction failed');

    const data = await response.json();
    updateUI(data);
  } catch (error) {
    console.error(error);
    alert('Failed to connect to the classification API. Make sure the backend is running.');
  } finally {
    loader.classList.add('hidden');
  }
}

function updateUI(data) {
  // Update main prediction
  topPrediction.textContent = data.prediction;
  topConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;

  // Update scores list
  scoresList.innerHTML = '';
  
  // Sort classes by score
  const sortedScores = Object.entries(data.all_scores)
    .sort(([, a], [, b]) => b - a);

  sortedScores.forEach(([className, score]) => {
    const scoreVal = (score * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'score-item';
    item.innerHTML = `
      <div class="score-info">
        <span class="class-name">${className}</span>
        <span class="score-percentage">${scoreVal}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: 0%"></div>
      </div>
    `;
    scoresList.appendChild(item);
    
    // Trigger animation
    setTimeout(() => {
      item.querySelector('.progress-fill').style.width = `${scoreVal}%`;
    }, 100);
  });

  // Switch cards
  uploadCard.classList.add('hidden');
  resultContainer.classList.remove('hidden');
}

function resetUI() {
  uploadCard.classList.remove('hidden');
  resultContainer.classList.add('hidden');
  fileInput.value = '';
}
