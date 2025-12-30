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
    window.switchMode = function(mode) {
        // Update the hidden Flask select element
        modeSelector.value = mode;

        // Update Tab styling
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        const activeTab = document.querySelector(`.tab[data-mode="${mode}"]`);
        if (activeTab) activeTab.classList.add('active');

        // Toggle Input Visibility
        if (mode === 'text') {
            inputTextGroup.classList.remove('hidden');
            inputFileGroup.classList.add('hidden');
            fileUpload.required = false; 
        } else {
            inputTextGroup.classList.add('hidden');
            inputFileGroup.classList.remove('hidden');
            fileUpload.required = true; 

            // Set file type filters
            if (mode === 'image') {
                fileUpload.setAttribute('accept', 'image/*');
                document.getElementById('file-label').innerText = "Upload Facial Image";
            } else if (mode === 'audio') {
                fileUpload.setAttribute('accept', 'audio/*');
                document.getElementById('file-label').innerText = "Upload Voice Recording";
            }
        }
        
        clearPreviews();
    };

    // --- 3. File Handling & Previews ---
    function clearPreviews() {
        previewContainer.classList.add('hidden');
        imagePreview.classList.add('hidden');
        audioPreview.classList.add('hidden');
        imagePreview.src = "";
        audioPreview.src = "";
        fileUpload.value = ""; 
        filenameDisplay.innerText = "";
    }

    fileUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        filenameDisplay.innerText = file.name;
        previewContainer.classList.remove('hidden');

        const mode = modeSelector.value;
        const reader = new FileReader();

        reader.onload = function(e) {
            if (mode === 'image') {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                audioPreview.classList.add('hidden');
            } else if (mode === 'audio') {
                audioPreview.src = e.target.result;
                audioPreview.classList.remove('hidden');
                imagePreview.classList.add('hidden');
            }
        };
        reader.readAsDataURL(file);
    });

    // --- 4. Drag and Drop ---
    if (dropZone) {
        dropZone.addEventListener("click", () => fileUpload.click());

        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.style.borderColor = "var(--primary)";
                dropZone.style.background = "#f0f4f5";
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.style.borderColor = "var(--border)";
                dropZone.style.background = "#fafbfc";
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileUpload.files = files;
                const event = new Event('change');
                fileUpload.dispatchEvent(event);
            }
        });
    }

    // --- 5. Form Submission & Error Prevention ---
    form.addEventListener("submit", (e) => {
        const currentMode = modeSelector.value;

        // Validation to prevent the "NoneType" error in Flask
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

        // Final safety check: if text is empty in Image/Audio mode,
        // send a single space " " instead of nothing to prevent Backend iteration errors.
        if (!textArea.value) {
            textArea.value = " ";
        }

        // Visual feedback
        loader.classList.remove("hidden");
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span>Processing...</span> <i class="fa-solid fa-circle-notch fa-spin"></i>`;
    });
});
