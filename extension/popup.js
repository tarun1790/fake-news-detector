document.addEventListener('DOMContentLoaded', () => {
    // State
    let serverUrl = 'http://localhost:8000';
    let currentScanData = null;

    // Elements
    const btnSettings = document.getElementById('btn-settings');
    const panelSettings = document.getElementById('panel-settings');
    const panelDashboard = document.getElementById('panel-dashboard');
    const panelLoader = document.getElementById('panel-loader');
    const panelResults = document.getElementById('panel-results');
    
    const serverUrlInput = document.getElementById('server-url');
    const btnSaveServer = document.getElementById('btn-save-server');
    const btnCancelSettings = document.getElementById('btn-cancel-settings');
    const serverErrorBanner = document.getElementById('server-error-banner');

    const btnScanPage = document.getElementById('btn-scan-page');
    const btnToggleManual = document.getElementById('btn-toggle-manual');
    const manualInputBox = document.getElementById('manual-input-box');
    const manualText = document.getElementById('manual-text');
    const btnScanManual = document.getElementById('btn-scan-manual');

    const loaderText = document.getElementById('loader-text');
    
    // Result elements
    const resultScore = document.getElementById('result-score');
    const scoreRing = document.getElementById('score-ring');
    const resultVerdict = document.getElementById('result-verdict');
    const resultMl = document.getElementById('result-ml');
    const resultLlm = document.getElementById('result-llm');
    const resultClickbait = document.getElementById('result-clickbait');
    const resultTitle = document.getElementById('result-title');
    const resultBias = document.getElementById('result-bias');
    const resultExplanation = document.getElementById('result-explanation');
    const resultFindings = document.getElementById('result-findings');
    
    const btnBack = document.getElementById('btn-back');
    const linkWebDashboard = document.getElementById('link-web-dashboard');

    // Load server url from storage
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
        chrome.storage.local.get(['serverUrl'], (result) => {
            if (result.serverUrl) {
                serverUrl = result.serverUrl;
                serverUrlInput.value = serverUrl;
            }
            testServerConnection();
        });
    } else {
        testServerConnection();
    }

    // --- NAVIGATION ---
    btnSettings.addEventListener('click', () => {
        showPanel(panelSettings);
    });

    btnCancelSettings.addEventListener('click', () => {
        showPanel(panelDashboard);
    });

    btnSaveServer.addEventListener('click', () => {
        const val = serverUrlInput.value.trim();
        if (!val) return;
        serverUrl = val;
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ serverUrl: serverUrl }, () => {
                showPanel(panelDashboard);
                testServerConnection();
            });
        } else {
            showPanel(panelDashboard);
            testServerConnection();
        }
    });

    btnToggleManual.addEventListener('click', () => {
        manualInputBox.classList.toggle('hidden');
        if (!manualInputBox.classList.contains('hidden')) {
            btnToggleManual.textContent = 'Hide manual input';
        } else {
            btnToggleManual.textContent = 'Or paste text manually';
        }
    });

    btnBack.addEventListener('click', () => {
        showPanel(panelDashboard);
        manualText.value = '';
        currentScanData = null;
    });

    linkWebDashboard.addEventListener('click', (e) => {
        e.preventDefault();
        if (typeof chrome !== 'undefined' && chrome.tabs) {
            chrome.tabs.create({ url: serverUrl });
        } else {
            window.open(serverUrl, '_blank');
        }
    });

    function showPanel(targetPanel) {
        [panelSettings, panelDashboard, panelLoader, panelResults].forEach(p => {
            if (p === targetPanel) p.classList.remove('hidden');
            else p.classList.add('hidden');
        });
    }

    // --- SERVER TEST ---
    async function testServerConnection() {
        try {
            const res = await fetch(`${serverUrl}/api/settings`, { method: 'GET', signal: AbortSignal.timeout(3000) });
            if (res.ok) {
                serverErrorBanner.classList.add('hidden');
                btnScanPage.disabled = false;
                btnScanManual.disabled = false;
                return true;
            }
        } catch (err) {
            console.warn('Luffy AI Backend offline:', err);
        }
        serverErrorBanner.classList.remove('hidden');
        btnScanPage.disabled = true;
        btnScanManual.disabled = true;
        return false;
    }

    // --- SCAN ACTIVE PAGE ---
    btnScanPage.addEventListener('click', async () => {
        const online = await testServerConnection();
        if (!online) {
            alert('Luffy AI Server is offline. Please launch the FastAPI server first.');
            return;
        }

        showPanel(panelLoader);
        loaderText.textContent = 'Connecting to active browser tab...';

        if (typeof chrome === 'undefined' || !chrome.tabs) {
            alert('This option is only available inside Chrome.');
            showPanel(panelDashboard);
            return;
        }

        try {
            // Get active tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) throw new Error('No active browser tab found.');

            // Check if page can be script-injected (avoid chrome:// or system pages)
            if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('edge://')) {
                throw new Error('Cannot scan system browser pages. Please open a news article.');
            }

            loaderText.textContent = 'Extracting article body content...';

            // Inject content.js script
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['content.js']
            });

            if (!results || !results[0] || !results[0].result) {
                throw new Error('Content script failed to extract page text.');
            }

            const pageData = results[0].result;
            if (!pageData.text || pageData.text.length < 50) {
                throw new Error('Article content is too short or page failed to load.');
            }

            loaderText.textContent = 'Analyzing stylistic patterns (ML)...';
            
            // Post payload to backend
            await sendToBackend(pageData);

        } catch (err) {
            console.error(err);
            alert(`Scanning Failed: ${err.message}`);
            showPanel(panelDashboard);
        }
    });

    // --- SCAN MANUAL TEXT ---
    btnScanManual.addEventListener('click', async () => {
        const text = manualText.value.trim();
        if (!text || text.length < 50) {
            alert('Please paste at least 50 characters of news text.');
            return;
        }

        const online = await testServerConnection();
        if (!online) {
            alert('Veritas AI Server is offline.');
            return;
        }

        showPanel(panelLoader);
        loaderText.textContent = 'Analyzing custom text stylistic markers...';

        try {
            await sendToBackend({
                title: 'Custom Paste Analysis',
                text: text,
                url: null
            });
        } catch (err) {
            alert(`Failed: ${err.message}`);
            showPanel(panelDashboard);
        }
    });

    async function sendToBackend(payload) {
        loaderText.textContent = 'Invoking AI Semantic Fact-Check...';
        
        try {
            const res = await fetch(`${serverUrl}/api/analyze/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: payload.title,
                    text: payload.text,
                    url: payload.url
                })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Analysis error');
            }

            const data = await res.json();
            currentScanData = data;
            displayResults(data);
        } catch (err) {
            throw new Error(`Server failed to analyze: ${err.message}`);
        }
    }

    function displayResults(data) {
        showPanel(panelResults);
        
        // Trust score
        const score = Math.round(data.trust_score);
        resultScore.textContent = score;

        // Render circular progress ring
        let color = '#10b981';
        let badgeClass = 'reliable';
        if (score < 30) {
            color = '#ef4444';
            badgeClass = 'fake';
        } else if (score < 60) {
            color = '#f59e0b';
            badgeClass = 'misleading';
        }

        scoreRing.style.stroke = color;
        const radius = scoreRing.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;
        scoreRing.style.strokeDasharray = `${circumference}`;
        scoreRing.style.strokeDashoffset = `${offset}`;

        // Verdict & Breakdown
        resultVerdict.textContent = data.verdict;
        resultVerdict.className = `badge ${badgeClass}`;
        
        resultMl.textContent = `${Math.round(data.ml_score)}%`;
        resultLlm.textContent = `${Math.round(data.llm_score)}%`;
        resultClickbait.textContent = `${data.clickbait_score}%`;
        
        resultTitle.textContent = data.title || 'Untitled Web Scan';
        resultBias.textContent = data.bias_rating;
        resultExplanation.textContent = data.explanation;

        // Key findings list
        resultFindings.innerHTML = '';
        data.key_findings.slice(0, 3).forEach(finding => {
            const li = document.createElement('li');
            li.textContent = finding;
            resultFindings.appendChild(li);
        });
    }
});
