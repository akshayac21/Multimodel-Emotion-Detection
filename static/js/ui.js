document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Element Selectors ---
    const modeSelector = document.getElementById("mode-selector");
    const inputTextGroup = document.getElementById("input-text");
    const inputFileGroup = document.getElementById("input-file");
    const textArea = document.querySelector('textarea[name="text"]');
    const fileUpload = document.getElementById("file-upload");
    const dropZone = document.getElementById("drop-zone");
    
    // Previews & Results
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const audioPreview = document.getElementById("audio-preview");
    const filenameDisplay = document.getElementById("filename-display");
    const plotImage = document.querySelector(".plot-image"); // The Gemini Result Plot
    
    // Form & UI
    const form = document.getElementById("analysisForm");
    const loader = document.getElementById("loader");
    const submitBtn = document.getElementById("submit-btn");

    // --- 2. Force Refresh Result Plot ---
    // This forces the browser to ignore the cache for the result image on page load
    if (plotImage && plotImage.src) {
        const baseUrl = plotImage.src.split('?')[0];
        plotImage.src = `${baseUrl}?t=${new Date().getTime()}`;
    }

    // --- 3. Tab / Mode Switching ---
    window.switchMode = function(mode) {
        modeSelector.value = mode;

        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        const activeTab = document.querySelector(`.tab[data-mode="${mode}"]`);
        if (activeTab) activeTab.classList.add('active');

        if (mode === 'text') {
            inputTextGroup.classList.remove("hidden");
            inputFileGroup.classList.add("hidden");
        } else {
            inputTextGroup.classList.add("hidden");
            inputFileGroup.classList.remove("hidden");
            
            const label = document.getElementById("file-label");
            if (label) {
                label.innerText = mode === 'image' ? "Upload Facial Expression (Image)" : "Upload Vocal Recording (Audio)";
            }
            fileUpload.setAttribute('accept', mode === 'image' ? 'image/*' : 'audio/*');
        }
    }

    // --- 4. File Upload & Preview Logic ---
    if (dropZone) {
        dropZone.addEventListener("click", () => fileUpload.click());
    }

    fileUpload.addEventListener("change", function() {
        const file = this.files[0];
        if (file) {
            previewContainer.classList.remove("hidden");
            filenameDisplay.innerText = `Selected: ${file.name}`;

            const reader = new FileReader();
            
            if (modeSelector.value === 'image' && file.type.startsWith("image/")) {
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove("hidden");
                    audioPreview.classList.add("hidden");
                };
                reader.readAsDataURL(file);
            } 
            else if (modeSelector.value === 'audio' && file.type.startsWith("audio/")) {
                const url = URL.createObjectURL(file);
                audioPreview.src = url;
                audioPreview.classList.remove("hidden");
                imagePreview.classList.add("hidden");
            }
        }
    });

    // --- 5. Drag and Drop Support ---
    if (dropZone) {
        ["dragover", "dragleave", "drop"].forEach(evt => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        dropZone.addEventListener("dragover", () => dropZone.classList.add("drop-zone--over"));
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drop-zone--over"));
        
        dropZone.addEventListener("drop", (e) => {
            dropZone.classList.remove("drop-zone--over");
            const files = e.dataTransfer.files;
            if (files.length) {
                fileUpload.files = files;
                fileUpload.dispatchEvent(new Event('change'));
            }
        });
    }

    // --- 6. Form Submission & Animation ---
    form.addEventListener("submit", (e) => {
        const currentMode = modeSelector.value;

        // Validation
        if (currentMode === 'text') {
            if (!textArea.value.trim()) {
                e.preventDefault();
                alert("Please enter a text statement for analysis.");
                return;
            }
        } else if (!fileUpload.files || fileUpload.files.length === 0) {
            e.preventDefault();
            alert(`Please upload an ${currentMode} file to proceed.`);
            return;
        }

        // Backend Safety
        if (!textArea.value) textArea.value = " ";

        // Show Loading UI
        loader.classList.remove("hidden");
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <span>AI is Analyzing...</span> 
            <i class="fa-solid fa-wand-magic-sparkles fa-beat" style="margin-left:8px;"></i>
        `;

        loader.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
});
