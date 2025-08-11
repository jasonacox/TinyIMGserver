// TinyIMGserver Web Interface JavaScript

class TinyIMGApp {
    constructor() {
        this.init();
        this.bindEvents();
        this.startStatusUpdater();
    }

    init() {
        // Initialize UI elements
        this.form = document.getElementById('generate-form');
        this.promptInput = document.getElementById('prompt');
        this.modelSelect = document.getElementById('model');
        this.sizeSelect = document.getElementById('size');
        this.stepsSlider = document.getElementById('steps');
        this.guidanceSlider = document.getElementById('guidance');
        this.seedInput = document.getElementById('seed');
        this.generateBtn = document.getElementById('generate-btn');
        this.progressContainer = document.getElementById('progress-container');
        this.imageContainer = document.getElementById('image-container');
        this.imageMetadata = document.getElementById('image-metadata');
        this.downloadContainer = document.getElementById('download-container');
        
        // Status elements
        this.statusIndicator = document.getElementById('status-indicator');
        this.totalRequests = document.getElementById('total-requests');
        this.queueLength = document.getElementById('queue-length');
        this.gpuCount = document.getElementById('gpu-count');
        
        // Slider value displays
        this.stepsValue = document.getElementById('steps-value');
        this.guidanceValue = document.getElementById('guidance-value');
        
        // Generated image data
        this.currentImageData = null;
    }

    bindEvents() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleGenerate(e));
        
        // Slider updates
        this.stepsSlider.addEventListener('input', (e) => {
            this.stepsValue.textContent = e.target.value;
        });
        
        this.guidanceSlider.addEventListener('input', (e) => {
            this.guidanceValue.textContent = e.target.value;
        });
        
        // Download button
        document.getElementById('download-btn').addEventListener('click', () => {
            this.downloadImage();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.handleGenerate(e);
            }
        });
    }

    async handleGenerate(e) {
        e.preventDefault();
        
        if (!this.promptInput.value.trim()) {
            this.showAlert('Please enter a prompt', 'warning');
            return;
        }
        
        try {
            this.setGenerating(true);
            
            const formData = this.getFormData();
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.displayImage(result);
            this.showAlert('Image generated successfully!', 'success');
            
        } catch (error) {
            console.error('Generation error:', error);
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.setGenerating(false);
        }
    }

    getFormData() {
        const [width, height] = this.sizeSelect.value.split('x').map(Number);
        
        const data = {
            prompt: this.promptInput.value.trim(),
            model: this.modelSelect.value,
            width: width,
            height: height,
            steps: parseInt(this.stepsSlider.value),
            guidance_scale: parseFloat(this.guidanceSlider.value)
        };
        
        // Only include seed if provided
        if (this.seedInput.value && this.seedInput.value.trim() !== '') {
            data.seed = parseInt(this.seedInput.value);
        }
        
        return data;
    }

    displayImage(result) {
        const { image, metadata } = result;
        
        // Create image element
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${image}`;
        img.alt = metadata.prompt;
        img.className = 'fade-in';
        
        // Update image container
        this.imageContainer.innerHTML = '';
        this.imageContainer.appendChild(img);
        
        // Update metadata
        document.getElementById('meta-model').textContent = metadata.model.toUpperCase();
        document.getElementById('meta-size').textContent = `${metadata.width}×${metadata.height}`;
        document.getElementById('meta-steps').textContent = metadata.steps;
        document.getElementById('meta-time').textContent = metadata.generation_time.toFixed(1);
        
        // Show metadata and download button
        this.imageMetadata.style.display = 'block';
        this.downloadContainer.style.display = 'block';
        
        // Store image data for download
        this.currentImageData = {
            data: image,
            filename: `tinyimg_${Date.now()}.png`
        };
    }

    downloadImage() {
        if (!this.currentImageData) return;
        
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${this.currentImageData.data}`;
        link.download = this.currentImageData.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        this.showAlert('Image downloaded!', 'info');
    }

    setGenerating(isGenerating) {
        if (isGenerating) {
            this.generateBtn.disabled = true;
            this.generateBtn.innerHTML = '<span class="loading-spinner me-2"></span>Generating...';
            this.progressContainer.style.display = 'block';
        } else {
            this.generateBtn.disabled = false;
            this.generateBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Image';
            this.progressContainer.style.display = 'none';
        }
    }

    showAlert(message, type = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }

    async updateStats() {
        try {
            const response = await fetch('/status');
            if (response.ok) {
                const status = await response.json();
                
                // Update status indicator
                this.statusIndicator.textContent = 'Online';
                this.statusIndicator.className = 'badge bg-success';
                
                // Update stats if available
                if (status.stats) {
                    this.totalRequests.textContent = status.stats.total_requests || 0;
                    this.queueLength.textContent = status.stats.queue_length || 0;
                }
            }
        } catch (error) {
            console.warn('Failed to update stats:', error);
            this.statusIndicator.textContent = 'Offline';
            this.statusIndicator.className = 'badge bg-danger';
        }
    }

    startStatusUpdater() {
        // Update stats every 5 seconds
        this.updateStats();
        setInterval(() => this.updateStats(), 5000);
    }

    // Utility methods
    formatTime(seconds) {
        return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showAlert('Copied to clipboard!', 'info');
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tinyImgApp = new TinyIMGApp();
});

// Example prompts for inspiration
const examplePrompts = [
    "A serene landscape with mountains and a lake at sunset",
    "A futuristic city with flying cars and neon lights",
    "A cute robot reading a book in a library",
    "An astronaut riding a horse on Mars",
    "A magical forest with glowing mushrooms and fireflies",
    "A steampunk locomotive flying through clouds",
    "A dragon made of crystalline formations",
    "A cozy coffee shop in a rainy city street"
];

// Add prompt suggestion functionality
function addPromptSuggestions() {
    const promptInput = document.getElementById('prompt');
    const suggestionsContainer = document.createElement('div');
    suggestionsContainer.className = 'mt-2';
    suggestionsContainer.innerHTML = `
        <small class="text-muted">
            <strong>Need inspiration?</strong> 
            <a href="#" id="random-prompt" class="text-decoration-none">Try a random prompt</a>
        </small>
    `;
    
    promptInput.parentNode.appendChild(suggestionsContainer);
    
    document.getElementById('random-prompt').addEventListener('click', (e) => {
        e.preventDefault();
        const randomPrompt = examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
        promptInput.value = randomPrompt;
        promptInput.focus();
    });
}

// Add prompt suggestions when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(addPromptSuggestions, 100);
});
