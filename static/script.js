document.addEventListener('DOMContentLoaded', () => {
    const pageContainer = document.getElementById('page-container');
    const loadingSpinner = document.getElementById('loading-spinner');

    // --- TEMPLATE CLONING ---
    const getTemplate = (id) => document.getElementById(id).content.cloneNode(true);

    // --- API CALLS ---
    const api = {
        analyze: async (formData) => {
            const response = await fetch('/api/analyze/', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Analysis request failed.');
            return response.json();
        },
        getPatients: async () => {
            const response = await fetch('/api/patients/');
            if (!response.ok) throw new Error('Failed to fetch patients.');
            return response.json();
        },
        getPatient: async (id) => {
            const response = await fetch(`/api/patients/${id}`);
            if (!response.ok) throw new Error('Failed to fetch patient details.');
            return response.json();
        }
    };

    // --- PAGE RENDERERS ---

    async function renderHomePage() {
        pageContainer.innerHTML = '';
        pageContainer.appendChild(getTemplate('home-page'));

        const analysisForm = document.getElementById('analysis-form');
        analysisForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(analysisForm);
            const submitBtn = document.getElementById('submit-btn');

            showLoading(true, submitBtn, 'Analyzing...');
            try {
                const patient = await api.analyze(formData);
                // Navigate to the new patient's detail page
                window.location.hash = `#/patients/${patient.id}`;
            } catch (error) {
                console.error(error);
                alert(error.message);
            } finally {
                showLoading(false, submitBtn, 'Analyze and Save');
            }
        });
    }

    async function renderPatientListPage() {
        showLoading(true);
        pageContainer.innerHTML = '';
        try {
            const patients = await api.getPatients();
            const page = getTemplate('patient-list-page');
            const container = page.getElementById('patients-container');

            if (patients.length === 0) {
                container.innerHTML = '<p>No patients found. Upload scans to get started.</p>';
            } else {
                patients.forEach(patient => {
                    const patientDiv = document.createElement('div');
                    patientDiv.className = 'patient-list-item';
                    patientDiv.innerHTML = `
                        <a href="#/patients/${patient.id}">
                            <strong>${escapeHTML(patient.name)}</strong>
                            <span>(${patient.images.length} image${patient.images.length === 1 ? '' : 's'})</span>
                        </a>
                    `;
                    container.appendChild(patientDiv);
                });
            }
            pageContainer.appendChild(page);
        } catch (error) {
            pageContainer.innerHTML = `<p class="error">${error.message}</p>`;
        } finally {
            showLoading(false);
        }
    }

    async function renderPatientDetailPage(id) {
        showLoading(true);
        pageContainer.innerHTML = '';
        try {
            const patient = await api.getPatient(id);
            const page = getTemplate('patient-detail-page');
            const resultsContainer = page.getElementById('results-container');

            page.getElementById('result-patient-name').textContent = patient.name;

            if (patient.images.length > 0) {
                patient.images.forEach(image => resultsContainer.appendChild(createResultCard(image)));
            } else {
                resultsContainer.innerHTML = '<p>No images found for this patient.</p>';
            }
            
            page.getElementById('back-to-patients-btn').addEventListener('click', () => {
                window.location.hash = '#/patients';
            });

            pageContainer.appendChild(page);
        } catch (error) {
            pageContainer.innerHTML = `<p class="error">${error.message}</p>`;
        } finally {
            showLoading(false);
        }
    }

    // --- ROUTER ---
    const routes = {
        '': renderHomePage,
        '#/patients': renderPatientListPage,
    };

    function router() {
        const hash = window.location.hash;
        const path = hash.split('/');
        
        let renderer;
        let param = null;

        if (path[1] === 'patients' && path[2]) { // Match #/patients/:id
            renderer = renderPatientDetailPage;
            param = path[2];
        } else { // Match '' or '#/patients'
            renderer = routes[hash] || routes[''];
        }
        
        renderer(param);
    }

    // --- HELPERS ---
    function showLoading(isLoading, btn = null, text = '') {
        loadingSpinner.classList.toggle('hidden', !isLoading);
        pageContainer.classList.toggle('hidden', isLoading);
        if (btn) {
            btn.disabled = isLoading;
            btn.textContent = text;
        }
    }

    function createResultCard(image) {
        const card = document.createElement('div');
        card.className = 'result-card';
        let content = '';

        if (image.analysis_workflow === 'error' || !image.annotated_image_url) {
            content = `
                <div class="result-info">
                    <h3>${escapeHTML(image.original_filename)}</h3>
                    <p class="error">Error: Analysis failed for this image.</p>
                </div>`;
        } else {
            content = `
                <img src="${image.annotated_image_url}" alt="Annotated image of ${escapeHTML(image.original_filename)}">
                <div class="result-info">
                    <p>${image.summary}</p>
                    <h3>${escapeHTML(image.original_filename)}</h3>
                    <p><strong>Workflow:</strong> <span class="workflow">${escapeHTML(image.analysis_workflow)}</span></p>
                    <p><strong>Confidence Scores:</strong> ${formatScores(image.scores)}</p>
                </div>`;
        }
        card.innerHTML = content;
        return card;
    }

    function formatScores(scores) {
        if (!scores || scores.length === 0) return 'N/A';
        return scores.map(s => s.toFixed(2)).join(', ');
    }

    function escapeHTML(str) {
        const p = document.createElement("p");
        p.textContent = str;
        return p.innerHTML;
    }

    // --- INITIALIZATION ---
    window.addEventListener('hashchange', router);
    router(); // Initial call to render the correct page on load
});
