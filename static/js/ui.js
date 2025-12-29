function changeMode(mode) {
    // 1. Update the hidden select field for Flask
    document.getElementById('mode-select').value = mode;

    // 2. Update Tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if(btn.innerText.toLowerCase() === mode) btn.classList.add('active');
    });

    // 3. Toggle visibility
    const textGroup = document.getElementById('text-input-group');
    const fileGroup = document.getElementById('file-input-group');

    if (mode === 'text') {
        textGroup.classList.remove('hidden');
        fileGroup.classList.add('hidden');
    } else {
        textGroup.classList.add('hidden');
        fileGroup.classList.remove('hidden');
    }
}

// Show filename after selection
document.getElementById('file-field').addEventListener('change', function(e) {
    const fileName = e.target.files[0] ? e.target.files[0].name : "Select File";
    document.getElementById('file-name').innerText = fileName;
});

// Loading state on submit
document.getElementById('mainForm').addEventListener('submit', function() {
    document.getElementById('loading').classList.remove('hidden');
});