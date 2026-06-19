document.addEventListener('DOMContentLoaded', () => {
    // --- LANDING SCREEN SCROLL REVEAL ---
    const btnScrollEnter = document.getElementById('btn-scroll-enter');
    const appContainer = document.querySelector('.app-container');
    
    if (btnScrollEnter && appContainer) {
        btnScrollEnter.addEventListener('click', () => {
            appContainer.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Scroll reveal observer
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                appContainer.classList.add('reveal-active');
            }
        });
    }, { threshold: 0.1 });

    if (appContainer) {
        revealObserver.observe(appContainer);
    }

    // Navigation / Tab state
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    
    // Scan settings/form buttons
    const scanTabBtns = document.querySelectorAll('.scan-tab-btn');
    const scanPanels = document.querySelectorAll('.scan-panel');
    const btnScanUrl = document.getElementById('btn-scan-url');
    const btnScanText = document.getElementById('btn-scan-text');
    const inputUrl = document.getElementById('input-url');
    const inputTitle = document.getElementById('input-title');
    const inputText = document.getElementById('input-text');
    
    // Scan Result controls
    const scanLoader = document.getElementById('scan-loader');
    const loaderStatus = document.getElementById('loader-status');
    const scanResults = document.getElementById('scan-results');
    const btnBackToScan = document.getElementById('btn-back-to-scan');
    
    // History log elements
    const historySearch = document.getElementById('history-search');
    const historyFilter = document.getElementById('history-filter');
    const historyTableBody = document.getElementById('history-table-body');
    const historyEmptyState = document.getElementById('history-empty-state');
    
    // Settings elements
    const settingGeminiKey = document.getElementById('setting-gemini-key');
    const btnToggleKeyVisibility = document.getElementById('btn-toggle-key-visibility');
    const keyVisibilityIcon = document.getElementById('key-visibility-icon');
    const keyStatusBadge = document.getElementById('key-status-badge');
    const btnSaveSettings = document.getElementById('btn-save-settings');
    const apiIndicator = document.getElementById('api-indicator');
    const apiStatusText = document.getElementById('api-status-text');
    const settingBackendUrl = document.getElementById('setting-backend-url');
    const btnSaveBackendUrl = document.getElementById('btn-save-backend-url');
    
    // --- HELPER TO GET DYNAMIC BACKEND API URL ---
    function getApiUrl(endpoint) {
        const storedUrl = localStorage.getItem('luffy_backend_url');
        const defaultBase = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1') ? '' : 'https://cuddly-dots-attend.loca.lt';
        const base = storedUrl || defaultBase;
        const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
        return `${cleanBase}${endpoint}`;
    }

    // --- API FETCH WRAPPER WITH BYPASS HEADER FOR LOCALTUNNEL ---
    async function apiFetch(endpoint, options = {}) {
        const url = getApiUrl(endpoint);
        options.headers = options.headers || {};
        options.headers['Bypass-Tunnel-Reminder'] = 'true';
        return fetch(url, options);
    }

    // Populate current backend URL
    if (settingBackendUrl) {
        settingBackendUrl.value = localStorage.getItem('luffy_backend_url') || 'https://cuddly-dots-attend.loca.lt';
    }

    if (btnSaveBackendUrl && settingBackendUrl) {
        btnSaveBackendUrl.addEventListener('click', () => {
            const url = settingBackendUrl.value.trim();
            if (!url) {
                localStorage.removeItem('luffy_backend_url');
                alert('Backend URL reset to default (localhost).');
                settingBackendUrl.value = 'http://localhost:8000';
            } else {
                localStorage.setItem('luffy_backend_url', url);
                alert(`Backend URL updated to: ${url}`);
            }
            checkSettingsStatus();
            loadDashboardData();
        });
    }
    
    // Modal controls
    const detailsModal = document.getElementById('details-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const modalContentBody = document.getElementById('modal-content-body');
    
    // Charts variables
    let timelineChart = null;
    let distributionChart = null;

    // Active state tracker
    let activeTab = 'dashboard';
    let scanMode = 'url'; // 'url' or 'text'
    let historyData = [];

    // --- TAB SYSTEM ---
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    function switchTab(tabId) {
        activeTab = tabId;
        
        // Update nav UI
        navItems.forEach(item => {
            if (item.getAttribute('data-tab') === tabId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Update content UI
        tabContents.forEach(tab => {
            if (tab.id === `tab-${tabId}`) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Header Title / Subtitle updates
        if (tabId === 'dashboard') {
            pageTitle.textContent = 'Dashboard Overview';
            pageSubtitle.textContent = 'Real-time media integrity metrics';
            loadDashboardData();
        } else if (tabId === 'analyze') {
            pageTitle.textContent = 'Scan Article';
            pageSubtitle.textContent = 'Linguistic stylistic audit and factual verification';
        } else if (tabId === 'history') {
            pageTitle.textContent = 'Scan History';
            pageSubtitle.textContent = 'Chronological log of verified claims';
            loadHistoryData();
        } else if (tabId === 'settings') {
            pageTitle.textContent = 'Configuration';
            pageSubtitle.textContent = 'Manage API connectivity and integrations';
            checkSettingsStatus();
        }
    }

    // --- SCAN FORM MODE SWITCH ---
    scanTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            scanMode = btn.getAttribute('data-scan-mode');
            
            scanTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            scanPanels.forEach(p => {
                if (p.id === `scan-form-${scanMode}`) {
                    p.classList.add('active');
                } else {
                    p.classList.remove('active');
                }
            });
        });
    });

    // --- SAVE SETTINGS (GEMINI KEY) ---
    btnToggleKeyVisibility.addEventListener('click', () => {
        if (settingGeminiKey.type === 'password') {
            settingGeminiKey.type = 'text';
            keyVisibilityIcon.classList.remove('fa-eye');
            keyVisibilityIcon.classList.add('fa-eye-slash');
        } else {
            settingGeminiKey.type = 'password';
            keyVisibilityIcon.classList.remove('fa-eye-slash');
            keyVisibilityIcon.classList.add('fa-eye');
        }
    });

    btnSaveSettings.addEventListener('click', async () => {
        const key = settingGeminiKey.value.trim();
        if (!key) {
            alert('Please enter a valid API key.');
            return;
        }

        try {
            btnSaveSettings.disabled = true;
            btnSaveSettings.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

            const res = await apiFetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gemini_api_key: key })
            });

            if (!res.ok) throw new Error('Failed to update settings');

            alert('Gemini API Key saved successfully!');
            settingGeminiKey.value = '';
            checkSettingsStatus();
        } catch (err) {
            console.error(err);
            alert('Error saving API Key.');
        } finally {
            btnSaveSettings.disabled = false;
            btnSaveSettings.innerHTML = '<i class="fa-solid fa-save"></i> Save API Key';
        }
    });

    const btnResetKey = document.getElementById('btn-reset-key');
    if (btnResetKey) {
        btnResetKey.addEventListener('click', () => {
            const setupForm = document.getElementById('settings-setup-form');
            const activeStatus = document.getElementById('settings-active-status');
            if (setupForm) setupForm.classList.remove('hidden');
            if (activeStatus) activeStatus.classList.add('hidden');
            settingGeminiKey.focus();
        });
    }

    async function checkSettingsStatus() {
        try {
            const res = await apiFetch('/api/settings');
            const data = await res.json();
            
            const setupForm = document.getElementById('settings-setup-form');
            const activeStatus = document.getElementById('settings-active-status');

            if (data.gemini_api_key_configured) {
                keyStatusBadge.textContent = 'Configured & Active';
                keyStatusBadge.className = 'key-status-badge configured';
                apiIndicator.className = 'status-indicator online';
                apiStatusText.textContent = 'AI Fact-Check Active';
                
                if (setupForm) setupForm.classList.add('hidden');
                if (activeStatus) activeStatus.classList.remove('hidden');
            } else {
                keyStatusBadge.textContent = 'Not Configured';
                keyStatusBadge.className = 'key-status-badge unconfigured';
                apiIndicator.className = 'status-indicator offline';
                apiStatusText.textContent = 'AI Fact-Check Disabled';
                
                if (setupForm) setupForm.classList.remove('hidden');
                if (activeStatus) activeStatus.classList.add('hidden');
            }
        } catch (err) {
            console.error('Error fetching settings status:', err);
        }
    }

    // --- SCAN LOGIC ---
    btnScanUrl.addEventListener('click', () => {
        const url = inputUrl.value.trim();
        if (!url) {
            alert('Please enter a valid URL.');
            return;
        }
        triggerScan('/api/analyze/url', { url: url });
    });

    btnScanText.addEventListener('click', () => {
        const title = inputTitle.value.trim();
        const text = inputText.value.trim();
        if (!text || text.length < 50) {
            alert('Please enter article text containing at least 50 characters.');
            return;
        }
        triggerScan('/api/analyze/text', { title: title || null, text: text });
    });

    async function triggerScan(endpoint, payload) {
        // Reset loader UI
        scanLoader.classList.remove('hidden');
        scanResults.classList.add('hidden');
        
        let progress = 15;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 1;
                document.getElementById('loader-progress').style.width = `${progress}%`;
                
                // Dynamic texts
                if (progress < 30) loaderStatus.textContent = 'Downloading article content...';
                else if (progress < 55) loaderStatus.textContent = 'Extracting linguistic stylistic structure...';
                else if (progress < 75) loaderStatus.textContent = 'Fact-checking claims via AI Analyzer...';
                else loaderStatus.textContent = 'Formulating credibility report...';
            }
        }, 500);

        try {
            const res = await apiFetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'An error occurred during scan.');
            }

            const data = await res.json();
            
            // Finish loader
            clearInterval(progressInterval);
            document.getElementById('loader-progress').style.width = '100%';
            
            setTimeout(() => {
                scanLoader.classList.add('hidden');
                displayScanResults(data);
            }, 500);

        } catch (err) {
            clearInterval(progressInterval);
            scanLoader.classList.add('hidden');
            alert(`Scanning Failed: ${err.message}`);
        }
    }

    function displayScanResults(data) {
        scanResults.classList.remove('hidden');
        
        // Title & Source url
        document.getElementById('result-title').textContent = data.title || 'Untitled Article';
        document.getElementById('result-source-url').textContent = data.url ? `Source: ${new URL(data.url).hostname}` : 'Source: Custom Paste';
        
        // Trust Score & Verdict badge
        const score = Math.round(data.trust_score);
        document.getElementById('result-trust-score').textContent = score;
        
        const verdictBadge = document.getElementById('result-verdict');
        verdictBadge.textContent = data.verdict;
        verdictBadge.className = 'verdict-badge';
        
        // Color settings
        let color = '#10b981'; // green
        if (score >= 60) {
            verdictBadge.classList.add('reliable');
            color = '#10b981';
        } else if (score >= 30) {
            verdictBadge.classList.add('misleading');
            color = '#f59e0b';
        } else {
            verdictBadge.classList.add('fake');
            color = '#ef4444';
        }

        // Animate circular progress ring
        const circle = document.getElementById('score-ring');
        circle.style.stroke = color;
        const radius = circle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;
        circle.style.strokeDasharray = `${circumference}`;
        circle.style.strokeDashoffset = `${offset}`;

        // Breakdown scores
        document.getElementById('result-ml-score').textContent = `${Math.round(data.ml_score)}%`;
        document.getElementById('result-llm-score').textContent = `${Math.round(data.llm_score)}%`;
        document.getElementById('result-clickbait-score').textContent = `${data.clickbait_score}%`;
        document.getElementById('result-explanation').textContent = data.explanation;

        // Key Findings
        const findingsList = document.getElementById('result-findings');
        findingsList.innerHTML = '';
        data.key_findings.forEach(finding => {
            const li = document.createElement('li');
            li.textContent = finding;
            findingsList.appendChild(li);
        });

        // Claim Verification
        const claimsList = document.getElementById('result-claims');
        claimsList.innerHTML = '';
        data.claims_analysed.forEach(claim => {
            const div = document.createElement('div');
            div.className = `claim-item ${claim.status}`;
            
            div.innerHTML = `
                <div class="claim-header">
                    <span class="claim-title">${claim.claim}</span>
                    <span class="claim-status-badge">${claim.status}</span>
                </div>
                <p class="claim-desc">${claim.explanation}</p>
            `;
            claimsList.appendChild(div);
        });
    }

    btnBackToScan.addEventListener('click', () => {
        scanResults.classList.add('hidden');
        inputUrl.value = '';
        inputTitle.value = '';
        inputText.value = '';
    });

    // --- DASHBOARD DATA LOADING ---
    async function loadDashboardData() {
        try {
            const res = await apiFetch('/api/stats');
            const data = await res.json();
            
            // Set simple numeric stats
            document.getElementById('metric-total-scans').textContent = data.total_scans;
            document.getElementById('metric-reliability').textContent = `${data.total_scans > 0 ? Math.round((data.reliable_count / data.total_scans) * 100) : 0}%`;
            document.getElementById('metric-clickbait').textContent = `${Math.round(data.avg_trust_score > 0 ? (100 - data.avg_trust_score) * 0.4 : 0)}%`; // Simulated clickbait average
            document.getElementById('metric-fake-count').textContent = data.fake_count + data.misleading_count;

            // Render Charts
            renderCharts(data);
        } catch (err) {
            console.error('Error loading dashboard analytics:', err);
        }
    }

    function renderCharts(data) {
        // Destroy existing instances if tabs reloaded
        if (timelineChart) timelineChart.destroy();
        if (distributionChart) distributionChart.destroy();

        // 1. Timeline Chart (Line/Area)
        const ctxTimeline = document.getElementById('timeline-chart').getContext('2d');
        const timelineLabels = data.timeline.map(item => item.date);
        const timelineCounts = data.timeline.map(item => item.count);
        const timelineScores = data.timeline.map(item => item.avg_score);

        timelineChart = new Chart(ctxTimeline, {
            type: 'line',
            data: {
                labels: timelineLabels,
                datasets: [
                    {
                        label: 'Articles Scanned',
                        data: timelineCounts,
                        borderColor: '#ffffff',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.3,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Average Trust %',
                        data: timelineScores,
                        borderColor: '#a1a1aa',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#a1a1aa', font: { family: 'SF Pro Text' } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#a1a1aa', font: { family: 'SF Pro Text' } }
                    },
                    y: {
                        position: 'left',
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#a1a1aa', stepSize: 1 }
                    },
                    y1: {
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#a1a1aa' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });

        // 2. Distribution Chart (Doughnut)
        const ctxDistribution = document.getElementById('distribution-chart').getContext('2d');
        distributionChart = new Chart(ctxDistribution, {
            type: 'doughnut',
            data: {
                labels: ['Reliable', 'Misleading', 'Fake'],
                datasets: [{
                    data: [data.reliable_count, data.misleading_count, data.fake_count],
                    backgroundColor: ['rgba(16, 185, 129, 0.75)', 'rgba(245, 158, 11, 0.75)', 'rgba(239, 68, 68, 0.75)'],
                    borderColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', font: { family: 'Outfit' } }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // --- HISTORY LOGIC ---
    async function loadHistoryData() {
        try {
            const res = await apiFetch('/api/history');
            historyData = await res.json();
            renderHistoryTable();
        } catch (err) {
            console.error('Error fetching history logs:', err);
        }
    }

    function renderHistoryTable() {
        const query = historySearch.value.toLowerCase().trim();
        const filter = historyFilter.value;
        
        let filtered = historyData.filter(item => {
            const matchesQuery = (item.title && item.title.toLowerCase().includes(query)) || 
                                 (item.text_content && item.text_content.toLowerCase().includes(query)) ||
                                 (item.url && item.url.toLowerCase().includes(query));
            
            let matchesFilter = true;
            if (filter !== 'all') {
                if (filter === 'Reliable') matchesFilter = item.trust_score >= 60;
                else if (filter === 'Mostly Reliable') matchesFilter = item.trust_score >= 60 && item.verdict.includes('Mostly');
                else if (filter === 'Plausible') matchesFilter = item.trust_score >= 45 && item.trust_score < 60;
                else if (filter === 'Misleading') matchesFilter = item.trust_score >= 25 && item.trust_score < 45;
                else if (filter === 'Fake') matchesFilter = item.trust_score < 25;
            }

            return matchesQuery && matchesFilter;
        });

        historyTableBody.innerHTML = '';
        
        if (filtered.length === 0) {
            historyEmptyState.classList.remove('hidden');
            return;
        }
        historyEmptyState.classList.add('hidden');

        filtered.forEach(item => {
            const tr = document.createElement('tr');
            
            // Format date
            const dateStr = new Date(item.created_at).toLocaleDateString(undefined, { 
                month: 'short', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            // Score color class
            let scoreColor = 'text-green';
            let badgeClass = 'reliable';
            if (item.trust_score < 30) {
                scoreColor = 'text-red';
                badgeClass = 'fake';
            } else if (item.trust_score < 60) {
                scoreColor = 'text-orange';
                badgeClass = 'misleading';
            }

            const titleCell = item.url ? 
                `<a href="${item.url}" target="_blank" rel="noopener">${item.title} <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:0.7rem; margin-left:4px;"></i></a>` : 
                `<span>${item.title}</span>`;

            tr.innerHTML = `
                <td>${dateStr}</td>
                <td>${titleCell}</td>
                <td>${item.bias_rating}</td>
                <td class="cell-score ${scoreColor}">${Math.round(item.trust_score)}%</td>
                <td><span class="badge-verdict ${badgeClass}">${item.verdict}</span></td>
                <td>
                    <div class="history-actions">
                        <button class="btn-icon-action hover-view" data-id="${item.id}" title="View Details">
                            <i class="fa-solid fa-eye"></i>
                        </button>
                        <button class="btn-icon-action hover-delete" data-id="${item.id}" title="Delete Record">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                </td>
            `;

            // Action Handlers
            tr.querySelector('.hover-view').addEventListener('click', () => showDetailsModal(item));
            tr.querySelector('.hover-delete').addEventListener('click', () => deleteHistoryItem(item.id));

            historyTableBody.appendChild(tr);
        });
    }

    historySearch.addEventListener('input', renderHistoryTable);
    historyFilter.addEventListener('change', renderHistoryTable);

    async function deleteHistoryItem(id) {
        if (!confirm('Are you sure you want to delete this scan log from your database?')) return;
        
        try {
            const res = await apiFetch(`/api/history/${id}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Delete failed');
            
            // Reload logs
            loadHistoryData();
        } catch (err) {
            console.error(err);
            alert('Error deleting historical scan log.');
        }
    }

    // --- MODAL LOGIC ---
    function showDetailsModal(item) {
        detailsModal.classList.remove('hidden');
        
        // Recreate the result layout inside the modal body
        let color = '#10b981';
        let badgeClass = 'reliable';
        if (item.trust_score < 30) {
            color = '#ef4444';
            badgeClass = 'fake';
        } else if (item.trust_score < 60) {
            color = '#f59e0b';
            badgeClass = 'misleading';
        }

        // Key findings markup
        const findingsMarkup = item.key_findings.map(finding => `<li>${finding}</li>`).join('');
        
        // Claims markup
        const claimsMarkup = item.claims_analysed.map(claim => `
            <div class="claim-item ${claim.status}">
                <div class="claim-header">
                    <span class="claim-title">${claim.claim}</span>
                    <span class="claim-status-badge">${claim.status}</span>
                </div>
                <p class="claim-desc">${claim.explanation}</p>
            </div>
        `).join('');

        modalContentBody.innerHTML = `
            <div class="results-header-row">
                <span class="source-tag">${item.url ? `Source: ${new URL(item.url).hostname}` : 'Source: Custom Paste'}</span>
                <span class="source-tag">Scan ID: #${item.id}</span>
            </div>
            
            <div class="results-summary-grid">
                <div class="result-score-card">
                    <h3>Trust Score</h3>
                    <div class="circular-progress-wrapper">
                        <svg class="progress-svg" viewBox="0 0 100 100">
                            <circle class="progress-bg" cx="50" cy="50" r="40"></circle>
                            <circle class="progress-fill" style="stroke: ${color}; stroke-dasharray: 251.2; stroke-dashoffset: ${251.2 - (item.trust_score/100)*251.2}" cx="50" cy="50" r="40"></circle>
                        </svg>
                        <div class="progress-score-text">
                            <span>${Math.round(item.trust_score)}</span>
                            <span class="percentage">%</span>
                        </div>
                    </div>
                    <div class="verdict-badge ${badgeClass}">${item.verdict}</div>
                </div>

                <div class="result-details-card">
                    <h3>${item.title}</h3>
                    <div class="scores-row">
                        <div class="score-pill">
                            <span class="score-pill-val">${Math.round(item.ml_score)}%</span>
                            <span class="score-pill-lbl">Linguistic Credibility</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-pill-val">${Math.round(item.llm_score)}%</span>
                            <span class="score-pill-lbl">Factual Consistency</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-pill-val">${item.clickbait_score}%</span>
                            <span class="score-pill-lbl">Clickbait Level</span>
                        </div>
                    </div>
                    <div class="verdict-narrative">
                        <h4>AI Analysis Narrative</h4>
                        <p>${item.explanation}</p>
                    </div>
                </div>
            </div>

            <div class="results-detailed-grid">
                <div class="details-card-block">
                    <h3><i class="fa-solid fa-list-check text-purple"></i>Key Findings</h3>
                    <ul class="findings-list">
                        ${findingsMarkup}
                    </ul>
                </div>
                <div class="details-card-block">
                    <h3><i class="fa-solid fa-clipboard-question text-blue"></i>Claim Verification</h3>
                    <div class="claims-list">
                        ${claimsMarkup}
                    </div>
                </div>
            </div>
        `;
    }

    btnCloseModal.addEventListener('click', () => {
        detailsModal.classList.add('hidden');
    });

    detailsModal.addEventListener('click', (e) => {
        if (e.target === detailsModal) detailsModal.classList.add('hidden');
    });

    // --- INITIALIZE ---
    checkSettingsStatus();
    loadDashboardData();
});
