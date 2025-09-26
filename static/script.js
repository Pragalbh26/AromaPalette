document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://127.0.0.1:5000';

    // --- Tab Handling ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.getAttribute('data-tab');

            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === tab) {
                    content.classList.add('active');
                }
            });
        });
    });

    // --- Single Molecule Analyzer ---
    const predictBtn = document.getElementById('predict-btn');
    const identifierInput = document.getElementById('molecule-identifier');
    const resultsContainer = document.getElementById('analyzer-results-container');
    const resultsDiv = document.getElementById('analyzer-results');
    const placeholderDiv = document.getElementById('analyzer-placeholder');
    const loader = document.getElementById('analyzer-loader');
    const predictionSpan = document.getElementById('result-prediction');
    const smilesSpan = document.getElementById('result-smiles');
    const imageEl = document.getElementById('molecule-image');

    predictBtn.addEventListener('click', async () => {
        const identifier = identifierInput.value.trim();
        if (!identifier) {
            alert('Please enter a molecule name or SMILES string.');
            return;
        }

        placeholderDiv.style.display = 'none';
        resultsContainer.style.display = 'flex';
        resultsDiv.style.display = 'none';
        loader.style.display = 'block';

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier })
            });
            const data = await response.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
                resultsContainer.style.display = 'none';
                placeholderDiv.style.display = 'flex';
            } else {
                predictionSpan.textContent = data.prediction;
                smilesSpan.textContent = data.smiles;
                imageEl.src = data.image;
                resultsDiv.style.display = 'block';
            }
        } catch (error) {
            console.error('Prediction failed:', error);
            alert('An error occurred. Check the console and make sure the Python backend is running.');
            resultsContainer.style.display = 'none';
            placeholderDiv.style.display = 'flex';
        } finally {
            loader.style.display = 'none';
        }
    });

    // --- Perfume Mixer ---
    const ingredientSelect = document.getElementById('ingredient-select');
    const blendList = document.getElementById('blend-list');
    const blendBtn = document.getElementById('blend-btn');
    const mixerResultsContainer = document.getElementById('mixer-results-container');
    const mixerResultsDiv = document.getElementById('mixer-results');
    const mixerPlaceholder = document.getElementById('mixer-placeholder');
    const mixerLoader = document.getElementById('mixer-loader');
    const dominantScentSpan = document.getElementById('dominant-scent');

    let currentBlend = new Set();

    async function populateIngredients() {
        try {
            const response = await fetch(`${API_BASE_URL}/ingredients`);
            const ingredients = await response.json();
            ingredients.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                ingredientSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load ingredients:', error);
        }
    }

    function renderBlendList() {
        blendList.innerHTML = '';
        currentBlend.forEach(ingredient => {
            const li = document.createElement('li');
            li.textContent = ingredient;
            const removeBtn = document.createElement('button');
            removeBtn.innerHTML = '&times;';
            removeBtn.className = 'remove-btn';
            removeBtn.onclick = () => {
                currentBlend.delete(ingredient);
                renderBlendList();
            };
            li.appendChild(removeBtn);
            blendList.appendChild(li);
        });
    }

    ingredientSelect.addEventListener('change', () => {
        const selected = ingredientSelect.value;
        if (selected && !currentBlend.has(selected)) {
            currentBlend.add(selected);
            renderBlendList();
        }
        ingredientSelect.value = ""; // Reset dropdown
    });
    
    blendBtn.addEventListener('click', async () => {
        if (currentBlend.size === 0) {
            alert('Add at least one ingredient to the blend.');
            return;
        }

        mixerPlaceholder.style.display = 'none';
        mixerResultsContainer.style.display = 'flex';
        mixerResultsDiv.style.display = 'none';
        mixerLoader.style.display = 'block';

        try {
            const response = await fetch(`${API_BASE_URL}/blend`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ingredients: Array.from(currentBlend) })
            });
            const data = await response.json();

            if(data.error) {
                alert(`Error: ${data.error}`);
            } else {
                 dominantScentSpan.textContent = data.dominant_scent;
                 mixerResultsDiv.style.display = 'block';
            }
        } catch (error) {
             console.error('Blending failed:', error);
             alert('An error occurred during blending. Check the console.');
             mixerResultsContainer.style.display = 'none';
             mixerPlaceholder.style.display = 'flex';
        } finally {
            mixerLoader.style.display = 'none';
        }
    });

    populateIngredients();
});

