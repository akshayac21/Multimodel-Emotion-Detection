document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Element Selectors ---
    const modeSelector = document.getElementById("mode-selector");
    const inputTextGroup = document.getElementById("input-text");
    const inputFileGroup = document.getElementById("input-file");
    const textArea = document.querySelector('textarea[name="text"]');
    const fileUpload = document.getElementById("file-upload");
    const dropZone = document.getElementById("drop-zone");
    
    // Previews
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const audioPreview = document.getElementById("audio-preview");
    const filenameDisplay = document.getElementById("filename-display");
    
    // Form & UI
    const form = document.getElementById("analysisForm");
    const loader = document.getElementById("loader");
    const submitBtn = document.getElementById("submit-btn");

    // --- 2. Tab / Mode Switching ---
    // This function is called by the onclick attributes in index.html
    window.switchMode = function(mode) {
        // Update the hidden input that Flask reads
        modeSelector.value = mode;

        // Update Tab UI Styling
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        const activeTab = document.querySelector(`.tab[data-mode="${mode}"]`);
        if (activeTab) activeTab.classList.add('active');

        // Toggle Input Visibility
        if (mode === 'text') {
            inputTextGroup.classList.remove("hidden");
            inputFileGroup.classList.add("hidden");
        } else {
            inputTextGroup.classList.add("hidden");
            inputFileGroup.classList.remove("hidden");
            
            // Update the text label inside the upload box based on mode
            const label = document.getElementById("file-label");
            if (label) {
                label.innerText = mode === 'image' ? "Upload Facial Expression (Image)" : "Upload Vocal Recording (Audio)";
            }
            
            // Dynamically update accepted file types
            fileUpload.setAttribute('accept', mode === 'image' ? 'image/*' : 'audio/*');
        }
    }

    // --- 3. File Upload Trigger & Preview ---
    // Make the entire drop zone clickable to trigger the hidden file input
    if (dropZone) {
        dropZone.addEventListener("click", () => {
            fileUpload.click();
        });
    }

    fileUpload.addEventListener("change", function() {
        const file = this.files[0];
        if (file) {
            previewContainer.classList.remove("hidden");
            filenameDisplay.innerText = `Selected: ${file.name}`;

            const reader = new FileReader();
            
            // Handle Image Preview
            if (modeSelector.value === 'image' && file.type.startsWith("image/")) {
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove("hidden");
                    audioPreview.classList.add("hidden");
                };
                reader.readAsDataURL(file);
            } 
            // Handle Audio Preview
            else if (modeSelector.value === 'audio' && file.type.startsWith("audio/")) {
                const url = URL.createObjectURL(file);
                audioPreview.src = url;
                audioPreview.classList.remove("hidden");
                imagePreview.classList.add("hidden");
            }
        }
    });

    // --- 4. Drag and Drop Support ---
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
                // Trigger the 'change' event manually so the preview logic runs
                const event = new Event('change');
                fileUpload.dispatchEvent(event);
            }
        });
    }

    // --- 5. Form Submission & Validation ---
    form.addEventListener("submit", (e) => {
        const currentMode = modeSelector.value;

        // Validation: Prevent submitting without required data
        if (currentMode === 'text') {
            if (!textArea.value.trim()) {
                e.preventDefault();
                alert("Please enter a text statement for analysis.");
                return;
            }
        } else {
            if (!fileUpload.files || fileUpload.files.length === 0) {
                e.preventDefault();
                alert(`Please upload an ${currentMode} file to proceed.`);
                return;
            }
        }

        // Backend Safety: Ensure 'text' field is never null for Image/Audio modes
        if (!textArea.value) {
            textArea.value = " ";
        }

        // Show Loading UI
        loader.classList.remove("hidden");
        submitBtn.disabled = true;
        
        // Give the user feedback that AI is working
        submitBtn.innerHTML = `
            <span>AI is Analyzing...</span> 
            <i class="fa-solid fa-wand-magic-sparkles fa-beat" style="margin-left:8px;"></i>
        `;

        // Auto-scroll to loader for better mobile UX
        loader.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
});
