document.addEventListener('DOMContentLoaded', () => {
    // Tab Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Static Tab Elements
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

    // Live Tab Elements
    const videoElem = document.getElementById('webcam-video');
    const canvasElem = document.getElementById('webcam-canvas');
    const liveOverlay = document.getElementById('live-overlay');
    const liveOverlayText = document.getElementById('live-overlay-text');
    const cameraStopBtn = document.getElementById('camera-stop-btn');
    
    const liveStatusCard = document.getElementById('live-status-card');
    const liveStatusIndicator = document.getElementById('live-status-indicator');
    const liveResultTitle = document.getElementById('live-result-title');
    const liveConfidenceBar = document.getElementById('live-confidence-bar');
    const liveConfidenceValue = document.getElementById('live-confidence-value');

    let videoStream = null;
    let analysisInterval = null;
    let isAnalyzingLive = false;

    // --- Tab Switching Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Handle Camera Lifecycle
            if (targetId === 'tab-live') {
                startCamera();
            } else {
                stopCamera();
            }
        });
    });

    // --- Static Upload Logic ---
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
        handleFiles(dt.files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    resetBtn.addEventListener('click', () => {
        resetStaticUI();
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.match('image.*')) {
            alert('Please select an image file (.jpg, .jpeg, .png).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            showAnalyzingStatic();
            uploadImage(file, false);
        };
        reader.readAsDataURL(file);
    }

    function showAnalyzingStatic() {
        uploadSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');
        
        resultTitle.textContent = "Analyzing Frame...";
        statusCard.className = 'status-card';
        statusIndicator.style.background = 'var(--text-secondary)';
        confidenceBar.style.width = '0%';
        confidenceBar.style.backgroundColor = 'var(--accent-blob-1)';
        confidenceValue.textContent = '0%';
    }

    function resetStaticUI() {
        uploadSection.classList.remove('hidden');
        resultSection.classList.add('hidden');
        fileInput.value = ''; 
        imagePreview.src = '';
    }

    // --- Live Camera Logic ---

    cameraStopBtn.addEventListener('click', stopCamera);

    async function startCamera() {
        if (videoStream) return; // Already running
        
        liveOverlay.classList.remove('hidden');
        liveOverlayText.textContent = "Requesting Camera Access...";
        cameraStopBtn.classList.remove('hidden');

        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: "environment" } // Prefer back camera
            });
            videoElem.srcObject = videoStream;
            
            videoElem.onloadedmetadata = () => {
                videoElem.play();
                liveOverlay.classList.add('hidden');
                
                // Set canvas size to match video internally
                canvasElem.width = videoElem.videoWidth;
                canvasElem.height = videoElem.videoHeight;
                
                // Start the 1-second analysis loop
                liveResultTitle.textContent = "Analyzing Stream...";
                analysisInterval = setInterval(captureAndAnalyzeWebcam, 1000);
            };
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            liveOverlayText.textContent = "Camera Error: " + err.message;
            liveResultTitle.textContent = "Camera Offline";
            cameraStopBtn.classList.add('hidden');
        }
    }

    function stopCamera() {
        if (analysisInterval) {
            clearInterval(analysisInterval);
            analysisInterval = null;
        }
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            videoElem.srcObject = null;
        }
        
        liveOverlay.classList.remove('hidden');
        liveOverlayText.textContent = "Camera Stopped";
        liveResultTitle.textContent = "Camera Offline";
        cameraStopBtn.classList.add('hidden');
        
        // Reset live UI cards
        liveStatusCard.className = 'status-card';
        liveConfidenceBar.style.width = '0%';
        liveConfidenceValue.textContent = '0%';
        liveConfidenceBar.style.backgroundColor = 'var(--accent-blob-1)';
    }

    async function captureAndAnalyzeWebcam() {
        // Prevent stacking requests if the network is extremely slow (>1s latency)
        if (isAnalyzingLive || !videoStream) return;
        
        const context = canvasElem.getContext('2d');
        // Redraw check to prevent black frames if metadata updated
        if(canvasElem.width !== videoElem.videoWidth) {
           canvasElem.width = videoElem.videoWidth;
           canvasElem.height = videoElem.videoHeight;
        }
        context.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

        isAnalyzingLive = true;

        canvasElem.toBlob((blob) => {
            if(!blob) {
                isAnalyzingLive = false;
                return;
            }
            // Model expects a file object
            const file = new File([blob], "live_frame.jpg", { type: "image/jpeg" });
            uploadImage(file, true);
        }, 'image/jpeg', 0.85); // 85% quality JPEG for speed over network
    }

    // --- Shared Core API Logic ---
    async function uploadImage(file, isLive = false) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network error');
            }

            const data = await response.json();
            
            if(isLive) {
                displayLiveResults(data);
            } else {
                displayStaticResults(data);
            }
        } catch (error) {
            console.error('Error Details:', error);
            if (!isLive) {
                alert('Error analyzing image. Ensure backend is running.');
                resetStaticUI();
            } else {
                // For live, fail silently to not interrupt the stream, just drop UI connection
                liveResultTitle.textContent = "Connection lost/retrying...";
            }
        } finally {
            if (!isLive) {
                loadingOverlay.classList.add('hidden');
            } else {
                isAnalyzingLive = false;
            }
        }
    }

    function displayStaticResults(data) {
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

    function displayLiveResults(data) {
        const { achieved, probability } = data;
        const probPct = (probability * 100).toFixed(1);

        liveConfidenceBar.style.width = `${probPct}%`;
        liveConfidenceValue.textContent = `${probPct}%`;
        
        if (achieved) {
            liveResultTitle.textContent = "CVS Achieved";
            liveStatusCard.classList.add('status-achieved');
            liveStatusCard.classList.remove('status-not-achieved');
            liveConfidenceBar.style.backgroundColor = 'var(--success-color)';
        } else {
            liveResultTitle.textContent = "CVS Not Achieved";
            liveStatusCard.classList.add('status-not-achieved');
            liveStatusCard.classList.remove('status-achieved');
            liveConfidenceBar.style.backgroundColor = 'var(--danger-color)';
        }
    }
});
