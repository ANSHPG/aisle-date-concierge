// ----------------------------------------------------------------------------
// APPLICATION STATE
// ----------------------------------------------------------------------------
let sessionId = null;
let appState = {
    chat_log: [],
    dossier: { city: null, weather: null, log: [] },
    pending_calls: [],
    location_saved: false,
    awaiting_location: false,
    pending_user_text: '',
    status: 'idle'
};
let isThinking = false;

// ----------------------------------------------------------------------------
// DOM ELEMENTS
// ----------------------------------------------------------------------------
const statusOpenAI = document.getElementById('status-openai');
const statusWeather = document.getElementById('status-weather');
const statusTavily = document.getElementById('status-tavily');

const dossierCity = document.getElementById('dossier-city');
const dossierWeatherSection = document.getElementById('dossier-weather-section');
const dossierTemp = document.getElementById('dossier-temp');
const dossierWeatherDesc = document.getElementById('dossier-weather-desc');
const dossierLog = document.getElementById('dossier-log');

const btnResetSession = document.getElementById('btn-reset-session');
const heroPanel = document.getElementById('hero-panel');
const chatViewport = document.getElementById('chat-viewport');
const chatHistory = document.getElementById('chat-history');
const actionCardsContainer = document.getElementById('action-cards-container');
const chatInput = document.getElementById('chat-input');
const btnSendMsg = document.getElementById('btn-send-msg');

// Mobile specific elements
const sidebarDossier = document.getElementById('sidebar-dossier');
const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
const btnCloseSidebar = document.getElementById('btn-close-sidebar');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const mobileCityTag = document.getElementById('mobile-city-tag');

// ----------------------------------------------------------------------------
// INITIALIZATION
// ----------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
    initSession();
    setupEventListeners();
});

function setupEventListeners() {
    // Input submit on Enter key
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitMessage();
        }
    });

    // Send button click
    btnSendMsg.addEventListener('click', submitMessage);

    // Reset button click
    btnResetSession.addEventListener('click', resetSession);

    // Quick chips click
    document.querySelectorAll('.qp-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt) {
                sendUserPayload(prompt);
            }
        });
    });

    // Delegated copy button listener
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.copy-btn');
        if (btn) {
            const text = btn.getAttribute('data-copy');
            if (text) {
                copyTextToClipboard(text, btn);
            }
        }
    });

    // Mobile Drawer Controls
    if (btnToggleSidebar) {
        btnToggleSidebar.addEventListener('click', openSidebarDrawer);
    }
    if (btnCloseSidebar) {
        btnCloseSidebar.addEventListener('click', closeSidebarDrawer);
    }
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', closeSidebarDrawer);
    }
}

function openSidebarDrawer() {
    if (sidebarDossier) sidebarDossier.classList.add('drawer-open');
    if (sidebarBackdrop) sidebarBackdrop.classList.add('active');
}

function closeSidebarDrawer() {
    if (sidebarDossier) sidebarDossier.classList.remove('drawer-open');
    if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
}



// ----------------------------------------------------------------------------
// API CALLS
// ----------------------------------------------------------------------------
async function initSession() {
    try {
        const response = await fetch('/api/session');
        const data = await response.json();
        sessionId = data.session_id;
        appState = data.state;
        
        updateStatusLights(data.system_status);
        updateUI();
    } catch (err) {
        console.error('Session initialization failed:', err);
    }
}

async function sendChatAction(action, extraPayload = {}) {
    if (!sessionId) return;
    
    isThinking = true;
    updateUI(); // renders the thinking spinner immediately

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                action: action,
                ...extraPayload
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        appState = data;
    } catch (err) {
        console.error('Chat transaction failed:', err);
        // Append error turn
        appState.chat_log.push({
            role: 'assistant',
            content: `Transaction error: ${err.message}. Please try again.`,
            structured: null
        });
    } finally {
        isThinking = false;
        updateUI();
    }
}

function sendUserPayload(text) {
    sendChatAction('message', { message: text });
}

function submitMessage() {
    const text = chatInput.value.trim();
    if (!text || isThinking || appState.status === 'awaiting_tool_approval') return;
    
    chatInput.value = '';
    closeSidebarDrawer();
    sendUserPayload(text);
}

function resetSession() {
    closeSidebarDrawer();
    sendChatAction('reset');
}

// ----------------------------------------------------------------------------
// CLINT-SIDE LOCATION ROUTINES (Fixes location failures in streamVersion2)
// ----------------------------------------------------------------------------
async function resolveLocationAutomatically() {
    const loaderText = document.getElementById('location-loader-text');
    if (loaderText) loaderText.innerText = 'Locating…';

    // 1. Try Browser Geolocation API
    if (navigator.geolocation) {
        try {
            const pos = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 8000 });
            });
            
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            
            // Reverse coordinates to City Name using OSM OpenStreetMap Nominatim API
            const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`);
            const data = await res.json();
            
            const city = data.address.city || data.address.town || data.address.village || data.address.county;
            if (city) {
                submitResolvedLocation(city);
                return;
            }
        } catch (err) {
            console.warn('Browser geolocation failed or rejected, falling back to IP geolocation.', err);
        }
    }

    // 2. Fallback to direct client-side IP-based geolocation (works in nested containers where geolocation is blocked)
    try {
        const res = await fetch('https://ipapi.co/json/');
        const data = await res.json();
        if (data.city) {
            submitResolvedLocation(data.city);
            return;
        }
    } catch (err) {
        console.warn('Client-side IP geolocation failed, falling back to server geoip lookup.', err);
    }

    // 3. Fallback to server-side geoip lookup
    try {
        const res = await fetch('/api/geoip');
        const data = await res.json();
        if (data.city) {
            submitResolvedLocation(data.city);
            return;
        }
    } catch (err) {
        console.error('All location discovery paths failed.', err);
    }

    // If all failed, show error message
    if (loaderText) {
        loaderText.innerText = 'Could not detect location. Please enter your city manually.';
    }
}

function submitResolvedLocation(city) {
    sendChatAction('resolve_location', { city: city });
}

function submitManualLocation() {
    const input = document.getElementById('manual-city-input');
    const city = input ? input.value.trim() : '';
    if (!city) return;
    submitResolvedLocation(city);
}

// Global hook for interactive scheduling prompt selections
window.submitOption = function(optionText) {
    if (isThinking) return;
    sendUserPayload(optionText);
};

// ----------------------------------------------------------------------------
// VIEW RENDERING
// ----------------------------------------------------------------------------
function updateStatusLights(status) {
    if (!status) return;
    
    statusOpenAI.className = 'dot ' + (status.openai ? 'dot-on' : 'dot-off');
    statusWeather.className = 'dot ' + (status.weather ? 'dot-on' : 'dot-off');
    statusTavily.className = 'dot ' + (status.tavily ? 'dot-on' : 'dot-off');
}

function updateUI() {
    // 1. Update Hero vs. Chat log viewports
    if (appState.chat_log.length === 0) {
        heroPanel.style.display = 'flex';
        chatViewport.style.display = 'none';
    } else {
        heroPanel.style.display = 'none';
        chatViewport.style.display = 'flex';
    }

    // 2. Update Dossier Metrics
    const cityText = appState.dossier.city || '—';
    dossierCity.innerText = cityText;
    if (mobileCityTag) {
        mobileCityTag.innerText = cityText;
    }
    
    if (appState.dossier.weather) {
        dossierWeatherSection.style.display = 'block';
        dossierTemp.innerText = `${appState.dossier.weather.temp}°C`;
        dossierWeatherDesc.innerText = `${appState.dossier.weather.desc} · feels ${appState.dossier.weather.feels_like}°C · ${appState.dossier.weather.humidity}% humidity · wind ${appState.dossier.weather.wind} m/s`;
    } else {
        dossierWeatherSection.style.display = 'none';
    }

    // 3. Update Dossier Tool logs
    dossierLog.innerHTML = '';
    if (appState.dossier.log.length === 0) {
        dossierLog.innerHTML = `<div class="dossier-sub empty-log">No tools called yet.</div>`;
    } else {
        // Render logs reversed (latest first)
        [...appState.dossier.log].reverse().forEach(entry => {
            const isOk = entry.status === 'approved';
            const logItem = document.createElement('div');
            logItem.className = `log-entry ${isOk ? 'log-approved' : 'log-denied'}`;
            logItem.innerHTML = `<b>${escapeHtml(entry.name)}</b><br>${escapeHtml(entry.status)} · ${entry.time}`;
            dossierLog.appendChild(logItem);
        });
    }

    // 4. Render Conversation history
    chatHistory.innerHTML = '';
    appState.chat_log.forEach((msg, idx) => {
        const isUser = msg.role === 'user';
        const msgRow = document.createElement('div');
        msgRow.className = `msg-row ${msg.role}`;
        
        const bubbleWrap = document.createElement('div');
        bubbleWrap.className = 'msg-bubble-wrap';
        
        const label = document.createElement('div');
        label.className = 'bubble-label';
        label.innerText = isUser ? 'You' : 'Aisle';
        
        const bubble = document.createElement('div');
        bubble.className = `bubble ${msg.role}`;
        
        if (isUser) {
            bubble.innerText = msg.content;
            bubbleWrap.appendChild(label);
            bubbleWrap.appendChild(bubble);
            
            // Add copy button
            const copyRow = document.createElement('div');
            copyRow.className = 'copy-row user';
            copyRow.innerHTML = `<button class="copy-btn" data-copy="${escapeAttr(msg.content)}">⧉ Copy</button>`;
            bubbleWrap.appendChild(copyRow);
        } else {
            // Assistant formatting (Structured / Conversational)
            const structured = msg.structured;
            if (structured) {
                // message part
                bubble.innerHTML = formatMessageText(structured.message || msg.content);
                bubbleWrap.appendChild(label);
                bubbleWrap.appendChild(bubble);
                
                // Render recommendations if present
                if (structured.restaurants && structured.restaurants.length > 0) {
                    const extra = document.createElement('div');
                    extra.className = 'assistant-extra';
                    extra.innerHTML = renderRestaurantsHTML(structured.restaurants);
                    bubbleWrap.appendChild(extra);
                }
                
                // Render planning timeline if present
                if (structured.plan) {
                    const extra = document.createElement('div');
                    extra.className = 'assistant-extra';
                    extra.innerHTML = renderPlannerHTML(structured.plan);
                    bubbleWrap.appendChild(extra);
                }
            } else {
                bubble.innerHTML = formatMessageText(msg.content);
                bubbleWrap.appendChild(label);
                bubbleWrap.appendChild(bubble);
            }
            
            // Add copy button matching raw formatted transcript payload
            const copyRow = document.createElement('div');
            copyRow.className = 'copy-row assistant';
            const rawCopyPayload = buildAssistantCopyText(msg);
            copyRow.innerHTML = `<button class="copy-btn" data-copy="${escapeAttr(rawCopyPayload)}">⧉ Copy</button>`;
            bubbleWrap.appendChild(copyRow);
        }
        
        msgRow.appendChild(bubbleWrap);
        chatHistory.appendChild(msgRow);
    });

    // 5. Render active action forms above text input
    actionCardsContainer.innerHTML = '';
    
    if (appState.status === 'awaiting_location') {
        const locCard = document.createElement('div');
        locCard.className = 'approval-card blue-frame';
        locCard.innerHTML = `
            <div class="approval-eyebrow">One quick thing</div>
            <div class="approval-title">Aisle would like your location</div>
            <div class="approval-args" id="location-loader-text">To recommend nearby restaurants, Aisle needs to know your city. Use your network location, or enter it manually below.</div>
            <div class="location-input-row">
                <input type="text" id="manual-city-input" placeholder="Enter city name (e.g. Rome, Mumbai)" autocomplete="off">
                <button class="btn btn-primary" onclick="submitManualLocation()">Proceed</button>
            </div>
            <div class="approval-actions" style="margin-top:0.75rem;">
                <button class="btn btn-secondary" onclick="resolveLocationAutomatically()">✦ Use my location</button>
            </div>
        `;
        actionCardsContainer.appendChild(locCard);
    } 
    else if (appState.status === 'awaiting_tool_approval') {
        const call = appState.pending_calls[0];
        const argsStr = Object.entries(call.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ');
        
        const approveCard = document.createElement('div');
        approveCard.className = 'approval-card';
        approveCard.innerHTML = `
            <div class="approval-eyebrow">Awaiting your approval</div>
            <div class="approval-title">Aisle wants to call <code>${escapeHtml(call.name)}</code></div>
            <div class="approval-args">${escapeHtml(argsStr || 'no arguments')}</div>
            <div class="approval-actions">
                <button class="btn btn-primary" onclick="sendChatAction('approve_tool')">✓ Approve</button>
                <button class="btn btn-secondary" onclick="sendChatAction('deny_tool')">✕ Deny</button>
            </div>
        `;
        actionCardsContainer.appendChild(approveCard);
    } 
    
    // Render Loader if backend task is compiling
    if (isThinking) {
        const loader = document.createElement('div');
        loader.className = 'thinking-log';
        loader.innerHTML = `
            <span style="font-size:16px;">✦</span>
            <span>· · · Aisle is thinking</span>
        `;
        actionCardsContainer.appendChild(loader);
        
        // Disable chat box while loading
        chatInput.disabled = true;
        btnSendMsg.style.opacity = 0.5;
        btnSendMsg.style.pointerEvents = 'none';
    } else {
        chatInput.disabled = false;
        btnSendMsg.style.opacity = 1;
        btnSendMsg.style.pointerEvents = 'auto';
    }

    // Scroll chat to viewport bottom
    setTimeout(() => {
        chatViewport.scrollTop = chatViewport.scrollHeight;
    }, 50);
}

// ----------------------------------------------------------------------------
// MARKDOWN / TEXT PARSERS (Handles User UI Box Formatting Requests)
// ----------------------------------------------------------------------------
function formatMessageText(text) {
    let html = escapeHtml(text);
    
    // Check if it's a schedule selector or choices layout asking for details
    const isDateQuestion = /date|time|schedule|when|day|availability|hour|calendar|vibe/i.test(text);
    // Matches bullet lists (- item) or numbered lists (1. item)
    const choiceMatch = text.match(/(?:\d+\.\s+|-\s+)(.+?)(?=\n(?:\d+\.\s+|-\s+)|$)/g);
    
    if (isDateQuestion && choiceMatch && choiceMatch.length > 0) {
        let choicesHtml = '';
        let mainText = text;
        
        choiceMatch.forEach(choice => {
            let cleanChoice = choice.replace(/^(?:\d+\.\s+|-\s+)/, '').trim();
            mainText = mainText.replace(choice, '');
            choicesHtml += `
                <div class="date-option-item" onclick="submitOption('${escapeAttr(cleanChoice)}')">
                    <span>${cleanChoice}</span>
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2.5" fill="none">
                        <polyline points="9 18 15 12 9 6"></polyline>
                    </svg>
                </div>
            `;
        });
        
        let formattedText = mainText.trim().replace(/\n/g, '<br>');
        if (!formattedText) formattedText = "Choose one of the preferred options:";
        
        return `
            <div class="regular-result-box">${formattedText}</div>
            <div class="date-prompt-box">
                <div class="date-prompt-header">
                    <span>📅</span> Scheduling Selector
                </div>
                <div class="date-prompt-content">Click an option below to proceed:</div>
                <div class="date-options-list">${choicesHtml}</div>
            </div>
        `;
    }
    
    // Frame standard conversational outputs nicely as requested
    return `<div class="regular-result-box">${html.replace(/\n/g, '<br>')}</div>`;
}

function renderRestaurantsHTML(restaurants) {
    const top = restaurants[0];
    const rest = restaurants.slice(1);
    
    let html = `
        <div class="rec-wrap">
            <div class="rec-heading">✦ Curated for you</div>
            <a class="rec-top-card" href="${top.link}" target="_blank" rel="noopener">
                <div class="rec-top-tag">Top Pick</div>
                <div class="rec-top-name">${escapeHtml(top.name)}</div>
                ${top.desc ? `<div class="rec-top-desc">${escapeHtml(top.desc)}</div>` : ''}
            </a>
    `;
    
    if (rest.length > 0) {
        html += `<div class="rec-grid">`;
        rest.forEach(r => {
            html += `
                <a class="rec-card" href="${r.link}" target="_blank" rel="noopener">
                    <div class="rec-card-name">${escapeHtml(r.name)}</div>
                    ${r.desc ? `<div class="rec-card-desc">${escapeHtml(r.desc)}</div>` : ''}
                </a>
            `;
        });
        html += `</div>`;
    }
    
    html += `</div>`;
    return html;
}

function renderPlannerHTML(plan) {
    const fields = [
        ['timeline', '🗺️', 'Best Date Plan'],
        ['best_area', '📍', 'Best Area / Location'],
        ['best_time', '⏰', 'Best Time'],
        ['things_to_avoid', '⚠️', 'Things To Avoid'],
        ['backup_plan', '🔁', 'Backup Plan']
    ];
    
    let html = `
        <div class="plan-wrap">
            <div class="plan-card">
                <div class="plan-heading">✦ Your Date Plan</div>
    `;
    
    fields.forEach(([key, icon, label]) => {
        const val = plan[key];
        if (!val) return;
        
        html += `
            <div class="plan-block">
                <div class="plan-label">${icon} ${label}</div>
        `;
        
        if (key === 'timeline') {
            html += renderTimelineHTML(val);
        } else {
            html += `<div class="plan-value">${escapeHtml(val)}</div>`;
        }
        
        html += `</div>`;
    });
    
    html += `</div></div>`;
    return html;
}

function renderTimelineHTML(timeline) {
    let items = [];
    
    if (Array.isArray(timeline)) {
        timeline.forEach(step => {
            let t = '', a = '';
            if (typeof step === 'object' && step !== null) {
                t = step.time || '';
                a = step.activity || step.description || '';
            } else {
                a = String(step);
            }
            if (!a) return;
            
            const timeHtml = t ? `<div class="timeline-time">${escapeHtml(t)}</div>` : '';
            items.push(`
                <div class="timeline-item">
                    <span class="timeline-dot"></span>
                    ${timeHtml}
                    <div class="timeline-activity">${escapeHtml(a)}</div>
                </div>
            `);
        });
    } else {
        // Fallback split parsing
        const text = String(timeline).trim();
        const pattern = /(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))/g;
        const pieces = text.split(pattern);
        
        if (pieces.length > 1) {
            let bufTime = '';
            pieces.forEach(piece => {
                const cleaned = piece.replace(/^[ -\u2013\u2014:]+/, '').trim();
                if (!cleaned) return;
                
                if (pattern.test(piece)) {
                    bufTime = piece;
                } else {
                    const timeHtml = bufTime ? `<div class="timeline-time">${escapeHtml(bufTime)}</div>` : '';
                    items.push(`
                        <div class="timeline-item">
                            <span class="timeline-dot"></span>
                            ${timeHtml}
                            <div class="timeline-activity">${escapeHtml(cleaned)}</div>
                        </div>
                    `);
                    bufTime = '';
                }
            });
        } else {
            items.push(`
                <div class="timeline-item">
                    <span class="timeline-dot"></span>
                    <div class="timeline-activity">${escapeHtml(text)}</div>
                </div>
            `);
        }
    }
    
    if (items.length === 0) return '';
    return `<div class="timeline-list">${items.join('')}</div>`;
}

// Recreate copy format builder
function buildAssistantCopyText(entry) {
    let parts = [entry.content || ''];
    const structured = entry.structured;
    
    if (structured) {
        const restaurants = structured.restaurants || [];
        if (restaurants.length > 0) {
            parts.push('\nRecommended places:');
            restaurants.forEach(r => {
                let line = `- ${r.name}`;
                if (r.desc) line += `: ${r.desc}`;
                parts.push(line);
            });
        }
        
        const plan = structured.plan;
        if (plan) {
            parts.push('\nDate plan:');
            if (plan.timeline) {
                parts.push('Best Date Plan:');
                if (Array.isArray(plan.timeline)) {
                    plan.timeline.forEach(step => {
                        if (typeof step === 'object' && step !== null) {
                            parts.push(`  ${step.time || ''} — ${step.activity || step.description || ''}`.trim());
                        } else {
                            parts.push(`  ${step}`);
                        }
                    });
                } else {
                    parts.push(`  ${plan.timeline}`);
                }
            }
            
            const fields = [
                ['best_area', 'Best Area / Location'],
                ['best_time', 'Best Time'],
                ['things_to_avoid', 'Things To Avoid'],
                ['backup_plan', 'Backup Plan']
            ];
            fields.forEach(([k, label]) => {
                if (plan[k]) parts.push(`${label}: ${plan[k]}`);
            });
        }
    }
    return parts.join('\n');
}

// ----------------------------------------------------------------------------
// UTILITY FUNCTIONS
// ----------------------------------------------------------------------------
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function escapeAttr(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function copyTextToClipboard(text, btn) {
    if (!navigator.clipboard) {
        // Fallback for non-https contexts
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {
            document.execCommand('copy');
            const orig = btn.innerText;
            btn.innerText = 'Copied ✓';
            setTimeout(() => btn.innerText = orig, 1200);
        } catch (err) {
            console.error('Fallback copy failed', err);
        }
        document.body.removeChild(textarea);
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        const orig = btn.innerText;
        btn.innerText = 'Copied ✓';
        setTimeout(() => btn.innerText = orig, 1200);
    }).catch(err => {
        console.error('Clipboard copy failed', err);
    });
}
