const API_URL = 'https://sunillamani03.pythonanywhere.com';
let selectedFile = null;
let currentFormat = 'image';

const formatConfig = {
    image: { accept: 'image/*', endpoint: '/api/analyze/image' },
    audio: { accept: 'audio/*', endpoint: '/api/analyze/audio' },
    video: { accept: 'video/*', endpoint: '/api/analyze/video' },
    pdf: { accept: '.pdf', endpoint: '/api/analyze/pdf' }
};

function selectFormat(format) {
    currentFormat = format;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('fileInput').accept = formatConfig[format].accept;
    reset();
}

document.getElementById('fileInput').addEventListener('change', function(e) {
    handleFile(e.target.files[0]);
});

const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); });
uploadArea.addEventListener('drop', (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const preview = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    
    if (currentFormat === 'image') {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    preview.style.display = 'block';
}

async function analyzeFile() {
    if (!selectedFile) { alert('Please select a file first'); return; }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch(API_URL + formatConfig[currentFormat].endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
}

function displayResults(result) {
    const container = document.getElementById('resultContainer');
    container.className = 'result-container ' + (result.isFake ? 'fake' : 'real');
    document.getElementById('resultTitle').textContent = result.isFake ? '⚠ Deepfake Detected!' : '✅ Appears Authentic';
    document.getElementById('confidence').textContent = 'Confidence: ' + result.confidence + '%';
    
    let html = '<h3>Analysis Details</h3>';
    for (let key in result.analyses) {
        html += '<div class="detail-item"><strong>' + key + ':</strong> ' + JSON.stringify(result.analyses[key]) + '</div>';
    }
    document.getElementById('details').innerHTML = html;
    container.style.display = 'block';
}

function reset() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('resultContainer').style.display = 'none';
}

console.log('App loaded. API:', API_URL);