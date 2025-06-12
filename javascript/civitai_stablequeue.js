// StableQueue integration for Civitai Randomizer
// Based on the working implementation from sd-civitai-browser-plus-stablequeue

// Helper function to update Gradio inputs
function updateInput(element) {
    let event = new Event('input', { bubbles: true });
    element.dispatchEvent(event);
}

// Function to find gradio app
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];
    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        }
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

// Sends prompt index to Python to extract metadata and queue in StableQueue
function sendToStableQueue(promptIndex) {
    console.log(`[CivitAI Randomizer StableQueue] Sending prompt ${promptIndex} to StableQueue`);
    
    const randomNumber = Math.floor(Math.random() * 1000);
    const paddedNumber = String(randomNumber).padStart(3, '0');
    const input = gradioApp().querySelector('#civitai_stablequeue_input textarea');
    
    if (!input) {
        console.error('[CivitAI Randomizer StableQueue] Could not find stablequeue input element. Make sure the extension is loaded.');
        alert('StableQueue integration not found. Please ensure the extension is properly installed.');
        return;
    }
    
    // Show loading feedback - find the button that was clicked
    const clickedButton = event?.target;
    const originalText = clickedButton?.textContent;
    if (clickedButton) {
        clickedButton.textContent = 'Sending to StableQueue...';
        clickedButton.style.opacity = '0.7';
        clickedButton.disabled = true;
    }
    
    // Send the prompt index with padding to the Python backend
    input.value = paddedNumber + "." + promptIndex;
    updateInput(input);
    
    // Reset button after a short delay
    setTimeout(() => {
        if (clickedButton) {
            clickedButton.textContent = originalText || '📤 Send to StableQueue';
            clickedButton.style.opacity = '1';
            clickedButton.disabled = false;
        }
    }, 3000);
}

console.log('[CivitAI Randomizer StableQueue] JavaScript loaded successfully'); 