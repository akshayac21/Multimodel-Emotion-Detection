document.addEventListener('DOMContentLoaded', () => {
    
    // Elements
    const modeSelect = document.getElementById('modeSelect');
    const textGroup = document.getElementById('textInputParams');
    const fileGroup = document.getElementById('fileInputParams');
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('fileName');
    const form = document.getElementById('analysisForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // 1. Handle Mode Switching
    function updateInputVisibility() {
        const mode = modeSelect.value;
        
        // Reset required attributes to prevent browser validation errors on hidden fields
        const textArea = document.getElementById('text');
        
        if (mode === 'text') {
            textGroup.classList.remove('hidden');
            fileGroup.classList.add('hidden');
            textArea.required = true;
            fileInput.required = false;
        } else {
            // Both Image and Audio use the file input
            textGroup.classList.add('hidden');
            fileGroup.classList.remove('hidden');
            textArea.required = false;
            fileInput.required = true;
            
            // Update accept attribute based on mode
            if (mode === 'image') {
                fileInput.accept = "image/*";
                fileNameDisplay.textContent = "Upload Facial Image (JPG, PNG)";
            } else {
                fileInput.accept = "audio/*";
                fileNameDisplay.textContent = "Upload Voice Recording (WAV, MP3)";
            }
        }
    }

    // Initialize state
    modeSelect.addEventListener('change', updateInputVisibility);
    updateInputVisibility(); // Run on load

    // 2. Enhance File Input UX
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = `Selected: ${e.target.files[0].name}`;
            fileNameDisplay.style.color = '#0F766E'; // Teal color indicating success
            fileNameDisplay.style.fontWeight = '600';
        }
    });

    // 3. Handle Form Submission & Loading
    form.addEventListener('submit', (e) => {
        // Basic Client-side Validation
        const mode = modeSelect.value;
        let isValid = true;

        if (mode === 'text' && !document.getElementById('text').value.trim()) {
            isValid = false;
            alert("Please enter text for analysis.");
        } else if (mode !== 'text' && fileInput.files.length === 0) {
            isValid = false;
            alert("Please select a file.");
        }

        if (isValid) {
            // Show Loader
            loadingOverlay.classList.remove('hidden');
            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Analyzing...';
        } else {
            e.preventDefault(); // Stop submission if invalid
        }
    });
});
