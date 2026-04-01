document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultSection = document.getElementById('result-section');
    const uploadSection = document.querySelector('.upload-section');
    const imagePreview = document.getElementById('image-preview');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');
    
    const statusCard = document.getElementById('status-card');
    const statusIndicator = document.getElementById('status-indicator');
    const resultTitle = document.getElementById('result-title');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    resetBtn.addEventListener('click', () => {
        resetUI();
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Basic image validation
        if (!file.type.match('image.*')) {
            alert('Please select an image file (.jpg, .jpeg, .png).');
            return;
        }

        // Show preview and loading state immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            showAnalyzing();
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    function showAnalyzing() {
        uploadSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');
        
        // Reset old results
        resultTitle.textContent = "Analyzing Frame...";
        statusCard.className = 'status-card';
        statusIndicator.style.background = 'var(--text-secondary)';
        confidenceBar.style.width = '0%';
        confidenceBar.style.backgroundColor = 'var(--accent-blob-1)';
        confidenceValue.textContent = '0%';
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Network error occurred');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing image: ' + error.message);
            resetUI();
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function displayResults(data) {
        const { achieved, probability } = data;
        const probPct = (probability * 100).toFixed(1);

        confidenceBar.style.width = `${probPct}%`;
        confidenceValue.textContent = `${probPct}%`;
        
        if (achieved) {
            resultTitle.textContent = "CVS Achieved";
            statusCard.classList.add('status-achieved');
            statusCard.classList.remove('status-not-achieved');
            confidenceBar.style.backgroundColor = 'var(--success-color)';
        } else {
            resultTitle.textContent = "CVS Not Achieved";
            statusCard.classList.add('status-not-achieved');
            statusCard.classList.remove('status-achieved');
            confidenceBar.style.backgroundColor = 'var(--danger-color)';
        }
    }

    function resetUI() {
        uploadSection.classList.remove('hidden');
        resultSection.classList.add('hidden');
        fileInput.value = ''; // Clear input
        imagePreview.src = '';
    }
});
