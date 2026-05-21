document.addEventListener('DOMContentLoaded', function() {
    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatMultilineText(text) {
        return escapeHtml(text).replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
    }
    
    const loadBtn = document.getElementById('loadBtn');
    const chapterList = document.getElementById('chapterList');
    const output = document.getElementById('output');
    const markerProfileSelect = document.getElementById('markerProfile');
    const scanBtn = document.getElementById('scanBtn');
    const ttsBtn = document.getElementById('ttsBtn');
    const currentFileEl = document.getElementById('currentFile');

    const mainVoiceSelect = document.getElementById('mainVoice');
    const footnoteVoiceSelect = document.getElementById('footnoteVoice');
    const testMainBtn = document.getElementById('testMainVoice');
    const testFootnoteBtn = document.getElementById('testFootnoteVoice');
    const footnoteModeSelect = document.getElementById('footnoteMode');
    const footnoteOptionsEl = document.getElementById('footnoteOptions');
    const textDisplayEl = document.getElementById('textDisplay');
    const playbackSpeedSelect = document.getElementById('playbackSpeed');

    const searchBarEl = document.getElementById('searchBar');
    const searchInputEl = document.getElementById('searchInput');
    const findBtn = document.getElementById('findBtn');
    const findPrevBtn = document.getElementById('findPrevBtn');
    const findNextBtn = document.getElementById('findNextBtn');
    const searchStatusEl = document.getElementById('searchStatus');
    const clearSearchBtn = document.getElementById('clearSearchBtn');

    let currentFilePath = null;
    let lastScanResults = null;
    let highlightInterval = null;
    let readingPositionMs = 0;
    let currentHighlightEl = null;
    let allSpans = [];
    let justSeeked = false;
    let isResumePlayback = false;

    let searchMatches = [];
    let currentMatchIdx = -1;

    const voiceGroups = [
        { label: "British Female", lang: "en-gb", voices: ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"] },
        { label: "British Male",   lang: "en-gb", voices: ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"] },
        { label: "American Female", lang: "en-us", voices: ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"] },
        { label: "American Male",  lang: "en-us", voices: ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"] },
        { label: "Other English",  lang: "en-gb", voices: ["ef_dora", "em_alex", "em_santa"] }
    ];

    const voiceLangMap = {};
    voiceGroups.forEach(group => {
        group.voices.forEach(v => { voiceLangMap[v] = group.lang; });
    });

    function populateVoiceDropdowns() {
        const mainVoice = localStorage.getItem('starlistener_mainVoice') || 'bf_emma';
        const footnoteVoice = localStorage.getItem('starlistener_footnoteVoice') || 'bm_george';

        [mainVoiceSelect, footnoteVoiceSelect].forEach(select => {
            select.innerHTML = '';
            voiceGroups.forEach(group => {
                const optgroup = document.createElement('optgroup');
                optgroup.label = group.label;
                group.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice;
                    option.textContent = voice;
                    optgroup.appendChild(option);
                });
                select.appendChild(optgroup);
            });
        });

        mainVoiceSelect.value = mainVoice;
        footnoteVoiceSelect.value = footnoteVoice;
    }

    populateVoiceDropdowns();

    mainVoiceSelect.addEventListener('change', () => {
        localStorage.setItem('starlistener_mainVoice', mainVoiceSelect.value);
    });

    footnoteVoiceSelect.addEventListener('change', () => {
        localStorage.setItem('starlistener_footnoteVoice', footnoteVoiceSelect.value);
    });

    footnoteModeSelect.addEventListener('change', () => {
        localStorage.setItem('starlistener_footnoteMode', footnoteModeSelect.value);
    });

    playbackSpeedSelect.addEventListener('change', () => {
        localStorage.setItem('starlistener_playbackSpeed', playbackSpeedSelect.value);
    });

    markerProfileSelect.addEventListener('change', () => {
        localStorage.setItem('starlistener_markerProfile', markerProfileSelect.value);
    });

    async function playVoiceTest(voice, label) {
        const lang = voiceLangMap[voice] || 'en-gb';
        const text = `Testing voice for Star Listener ${label} voice ${voice}.`;
        const payload = JSON.stringify({ text, voice, speed: 1.0, lang });

        output.innerText = `Generating test for ${voice}...`;

        try {
            const response = await window.electronAPI.runTTS(payload);
            const result = JSON.parse(response);
            if (result.error) {
                output.innerText = `Test error: ${result.error}`;
                return;
            }

            const fileUrl = `file:///${result.output_path.replace(/\\/g, '/')}`;
            const existing = document.getElementById('testAudio');
            if (existing) existing.remove();

            const audio = document.createElement('audio');
            audio.id = 'testAudio';
            audio.controls = true;
            audio.src = fileUrl;
            audio.autoplay = true;
            audio.style.cssText = 'margin-top: 8px; width: 100%; max-width: 400px;';
            const voiceSection = document.querySelector('.voice-selectors');
            if (voiceSection) voiceSection.appendChild(audio);

            output.innerText = `Test ready (${result.duration_s}s) — playing`;
        } catch (e) {
            const msg = (e && e.message) ? e.message : String(e);
            output.innerText = `Test error: ${msg}`;
        }
    }

    testMainBtn.addEventListener('click', () => {
        playVoiceTest(mainVoiceSelect.value, 'main prose');
    });

    testFootnoteBtn.addEventListener('click', () => {
        playVoiceTest(footnoteVoiceSelect.value, 'footnotes');
    });

    const savedSpeed = localStorage.getItem('starlistener_playbackSpeed');
    if (savedSpeed) playbackSpeedSelect.value = savedSpeed;

    scanBtn.addEventListener('click', async () => {
        if (!currentFilePath) return;

        const marker_profile = markerProfileSelect ? markerProfileSelect.value : 'auto_heur';
        const payload = JSON.stringify({
            path: currentFilePath,
            options: { marker_profile }
        });

        output.innerText = `Scanning… (${marker_profile})`;
        scanBtn.disabled = true;

        try {
            const response = await window.electronAPI.runPython(payload);
            const parsed = JSON.parse(response);
            lastScanResults = Array.isArray(parsed) ? parsed : (parsed.rows || []);
            if (lastScanResults.length > 0) {
                if (currentFilePath) {
                    const fname = currentFilePath.split(/[/\\]/).pop();
                    const safeFolder = fname.replace(/\s+/g, '_');
                    const stem = fname.replace(/\.[^.]+$/, '').replace(/\s+/g, '_');
                    const nmPath = `output\\${safeFolder}\\${stem}_notemarkers.json`;
                    window.electronAPI.writeJsonFile(nmPath, lastScanResults);
                    window.electronAPI.saveSessionInfo({ bookPath: currentFilePath });
                }
                footnoteOptionsEl.style.display = 'flex';
                footnoteModeSelect.disabled = false;
            } else {
                footnoteOptionsEl.style.display = 'none';
                footnoteModeSelect.disabled = true;
            }
            showFootnotes(response);
        } catch (e) {
            const msg = (e && e.message) ? e.message : String(e);
            output.innerText = `Engine error: ${msg}`;
            chapterList.innerText = '';
            const container = document.getElementById('footnoteContainer');
            if (container) container.innerHTML = '';
        } finally {
            scanBtn.disabled = false;
        }
    });

    ttsBtn.addEventListener('click', async () => {
        if (!currentFilePath) {
            output.innerText = "No book loaded. Open a file first.";
            return;
        }

        const voice = mainVoiceSelect.value || 'bf_emma';
        const lang = voiceLangMap[voice] || 'en-gb';
        const footnoteMode = (!footnoteModeSelect.disabled && lastScanResults && lastScanResults.length > 0)
            ? footnoteModeSelect.value
            : 'as_is';
        const payload = JSON.stringify({
            epub_path: currentFilePath,
            voice: voice,
            footnote_voice: footnoteVoiceSelect.value || 'bm_george',
            speed: parseFloat(playbackSpeedSelect.value) || 1.0,
            lang: lang,
            footnote_mode: footnoteMode,
            footnotes: footnoteMode !== 'as_is' ? lastScanResults : undefined
        });

        output.innerText = "Extracting text...";
        ttsBtn.disabled = true;

        readingPositionMs = 0;
        currentHighlightEl = null;
        allSpans = [];
        const tb = document.getElementById('transportBar');
        if (tb) tb.style.display = 'none';

        let statusBase = '';
        const statusHandler = (data) => {
            if (data && data.status && data.value) {
                if (data.status === 'paragraph') {
                    output.innerText = statusBase ? `${statusBase} \u2014 Processing ${data.value}` : `Processing ${data.value}`;
                } else {
                    statusBase = `${data.status}: ${data.value}`;
                    output.innerText = statusBase;
                }
            }
        };
        window.electronAPI.onTtsStatus(statusHandler);

        try {
            const response = await window.electronAPI.runTTS(payload);
            const result = JSON.parse(response);
            if (result.error) {
                output.innerText = `TTS error: ${result.error}`;
                return;
            }

            const fileUrl = `file:///${result.output_path.replace(/\\/g, '/')}`;
            const existingAudio = document.getElementById('ttsAudio');
            if (existingAudio) existingAudio.remove();

            const audio = document.createElement('audio');
            audio.id = 'ttsAudio';
            audio.controls = true;
            audio.src = fileUrl;
            audio.style.cssText = 'margin-top: 12px; width: 100%;';
            document.getElementById('audioContainer').appendChild(audio);

            if (result.paragraph_timestamps && result.paragraph_timestamps.length > 0) {
                renderTextDisplay(result.full_text, result.word_timestamps, result.paragraph_timestamps);
                startHighlighting(audio, result.paragraph_timestamps);
            } else if (result.word_timestamps && result.word_timestamps.length > 0) {
                renderTextDisplay(result.full_text, result.word_timestamps);
                startHighlighting(audio, result.word_timestamps);
            }

            const wc = result.word_count ? `, ${result.word_count} words` : '';
            output.innerText = `Speech generated (${result.duration_s}s, ${result.sample_rate}Hz${wc})`;

            if (currentFilePath) {
                const sessionData = {
                    bookPath: currentFilePath,
                    audioPath: result.output_path || '',
                    timestampsPath: result.timestamps_path || '',
                    readingPositionMs: readingPositionMs || 0,
                };
                window.electronAPI.saveSessionInfo(sessionData);
            }
        } catch (e) {
            const msg = (e && e.message) ? e.message : String(e);
            output.innerText = `TTS error: ${msg}`;
        } finally {
            ttsBtn.disabled = false;
        }
    });

    loadBtn.addEventListener('click', async () => {
        const filePath = await window.electronAPI.openFile();
        if (filePath) {
            currentFilePath = filePath;
            lastScanResults = null;
            document.getElementById('footnoteContainer').innerHTML = '';
            chapterList.innerHTML = '';
            output.innerText = '';
            footnoteOptionsEl.style.display = 'none';
            footnoteModeSelect.disabled = true;
            footnoteModeSelect.value = 'as_is';
            textDisplayEl.style.display = 'none';
            textDisplayEl.innerHTML = '';
            searchBarEl.style.display = 'none';
            clearSearchHighlights();
            searchInputEl.value = '';
            readingPositionMs = 0;
            currentHighlightEl = null;
            allSpans = [];
            const tb = document.getElementById('transportBar');
            if (tb) tb.style.display = 'none';
            const existingAudio = document.getElementById('ttsAudio');
            if (existingAudio) {
                if (highlightInterval) {
                    existingAudio.removeEventListener('timeupdate', highlightInterval);
                    highlightInterval = null;
                }
                existingAudio.remove();
            }
            const basename = filePath.split(/[/\\]/).pop();
            currentFileEl.innerText = basename;
            window.electronAPI.saveSessionInfo({ bookPath: currentFilePath });
            scanBtn.disabled = false;
            ttsBtn.disabled = false;
            loadBookData(currentFilePath);
        }
    });

    async function showFootnotes(data) {
        console.log("Raw data received:", data);
        let matches;
        try {
            matches = JSON.parse(data);
        } catch (e) {
            output.innerText = `Failed to parse engine output: ${e && e.message ? e.message : e}`;
            // Still show raw output to help debug.
            chapterList.innerText = String(data || '');
            return;
        }
        console.log("Parsed matches:", matches);
        console.log("Number of matches:", matches.length);

        // Show raw JSON in an expandable block (useful for debugging regressions).
        chapterList.innerHTML = '';
        const details = document.createElement('details');
        details.style.marginTop = '12px';
        const summary = document.createElement('summary');
        summary.innerText = 'Raw JSON (engine output)';
        const pre = document.createElement('pre');
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.maxHeight = '240px';
        pre.style.overflow = 'auto';
        pre.textContent = JSON.stringify(matches, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        chapterList.appendChild(details);

        const aiNeeded = matches.filter(x => x && x.match_method === 'ai').length;
        const aiMatched = matches.filter(x => x && x.ai_status === 'matched').length;
        const aiUnavailable = matches.filter(x => x && x.ai_status === 'unavailable').length;
        const profileUsed = (matches.find(x => x && x.marker_profile) || {}).marker_profile;

        const summaryParts = [];
        if (profileUsed) summaryParts.push(`profile=${profileUsed}`);
        if (aiNeeded > 0) summaryParts.push(`AI-needed=${aiNeeded}`);
        if (aiMatched > 0) summaryParts.push(`AI-matched=${aiMatched}`);
        if (aiUnavailable > 0) summaryParts.push(`AI-unavailable=${aiUnavailable}`);
        output.innerText = summaryParts.length ? summaryParts.join(' • ') : `Found ${matches.length} items`;
        const container = document.getElementById('footnoteContainer');
        container.innerHTML = '';

        // Add a "Confirm All" button at the top
        const topConfirmBtn = document.createElement('button');
        topConfirmBtn.innerText = 'Confirm All Footnotes';
        topConfirmBtn.style.cssText = 'background: #2e7d32; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 20px; font-size: 1em;';
        topConfirmBtn.addEventListener('click', function() {
            const cards = container.querySelectorAll('.footnote-card');
            cards.forEach(card => {
                card.style.opacity = '0.3';
                card.style.pointerEvents = 'none';
            });
        });
        container.appendChild(topConfirmBtn);

        const getChapterKey = (item) => {
            if (item.source === 'epub') {
                return `epub:${item.chapter_group || item.chapter_label || item.chapter_index}`;
            }
            if (item.source === 'pdf') return `pdf:${item.page_index}`;
            if (item.source === 'text') return `text:${item.file_name || ''}`;
            return `other:${item.source || ''}`;
        };

        const getChapterLabel = (item) => {
            if (item.source === 'epub') {
                const idx = Number.isFinite(Number(item.chapter_index)) ? Number(item.chapter_index) : item.chapter_index;
                if (item.chapter_label) return String(item.chapter_label);
                const name = item.chapter_name ? String(item.chapter_name).split('/').pop() : `Item ${idx}`;
                return `Spine ${idx}: ${name}`;
            }
            if (item.source === 'pdf') return `Page ${Number(item.page_index) + 1}`;
            if (item.source === 'text') return `File: ${item.file_name || 'Text'}`;
            return `${item.source || 'Source'} `;
        };

        const getBestChapterLabel = (items) => {
            if (!items || !items.length) return '';
            const first = items[0];
            if (!first || first.source !== 'epub') return getChapterLabel(first || {});

            // Prefer a label within the group that actually contains a chapter token
            // (Roman/Arabic), since items[0] might be an early hit before the
            // numeral-bearing heading is encountered.
            const tokenCounts = new Map();
            const tokenFirstIndex = new Map();
            const tokenBestLabel = new Map();

            for (let i = 0; i < items.length; i++) {
                const it = items[i];
                const lab = it && it.chapter_label ? String(it.chapter_label) : '';
                if (!lab) continue;
                const tok = tryExtractChapterToken(lab);
                if (tok == null) continue;
                const key = String(tok);
                tokenCounts.set(key, (tokenCounts.get(key) || 0) + 1);
                if (!tokenFirstIndex.has(key)) tokenFirstIndex.set(key, i);
                const prev = tokenBestLabel.get(key);
                if (!prev || lab.length > prev.length) tokenBestLabel.set(key, lab);
            }

            if (tokenCounts.size) {
                // Pick the most common token; break ties by earliest appearance.
                let bestKey = null;
                let bestCount = -1;
                let bestFirst = Number.POSITIVE_INFINITY;
                for (const [k, c] of tokenCounts.entries()) {
                    const f = tokenFirstIndex.get(k) ?? Number.POSITIVE_INFINITY;
                    if (c > bestCount || (c === bestCount && f < bestFirst)) {
                        bestKey = k;
                        bestCount = c;
                        bestFirst = f;
                    }
                }
                const bestLab = bestKey != null ? tokenBestLabel.get(bestKey) : null;
                if (bestLab) return bestLab;
            }

            // Fall back to the first available chapter_label, then to the default label.
            for (const it of items) {
                if (it && it.chapter_label) return String(it.chapter_label);
            }
            return getChapterLabel(first);
        };

        const getOrderValue = (item) => {
            const ok = typeof item.order_key === 'number' ? item.order_key : -1;
            if (ok >= 0) return ok;
            const pos = typeof item.position === 'number' ? item.position : -1;
            if (pos >= 0) return pos;
            return typeof item.id === 'number' ? item.id : 0;
        };

        const romanToInt = (s) => {
            const t = String(s || '').toUpperCase().replace(/[^IVXLC]/g, '');
            if (!t) return null;
            const vals = { I: 1, V: 5, X: 10, L: 50, C: 100 };
            let total = 0;
            let prev = 0;
            for (let i = t.length - 1; i >= 0; i--) {
                const ch = t[i];
                const v = vals[ch];
                if (!v) return null;
                if (v < prev) total -= v;
                else {
                    total += v;
                    prev = v;
                }
            }
            if (total <= 0 || total > 500) return null;
            return total;
        };

        const tryExtractChapterToken = (txt) => {
            const t = String(txt || '').trim();
            if (!t) return null;

            // Avoid treating PART headings as chapter numbers.
            if (/^\s*\[?\s*PART\b/i.test(t)) return null;

            // Matches:
            //   CHAPTER VI
            //   NOTES TO CHAPTER VI
            //   NOTES ON CHAPTER 6
            let m = t.match(/\b(?:NOTES|FOOTNOTES|ENDNOTES)\s+(?:TO|ON)\s+CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b/i);
            if (!m) m = t.match(/\bCHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b/i);
            if (!m) m = t.match(/^\s*([IVXLC]{1,12}|\d{1,3})\s*\./i);
            if (!m) m = t.match(/^\s*([IVXLC]{1,12}|\d{1,3})\b/i);
            if (!m) return null;

            const token = m[1];
            if (!token) return null;
            if (/^\d{1,3}$/.test(token)) {
                const n = Number(token);
                return Number.isFinite(n) ? n : null;
            }
            return romanToInt(token);
        };

        const minGroupOrder = (items) => {
            if (!items || !items.length) return Number.POSITIVE_INFINITY;
            let best = Number.POSITIVE_INFINITY;
            for (const it of items) {
                const v = getOrderValue(it);
                if (typeof v === 'number' && Number.isFinite(v)) best = Math.min(best, v);
            }
            return best;
        };

        // Group matches by chapter/page
        const chapters = {};
        matches.forEach(item => {
            const key = getChapterKey(item);
            if (!chapters[key]) chapters[key] = [];
            chapters[key].push(item);
        });

        const chapterKeys = Object.keys(chapters).sort((a, b) => {
            const aItems = chapters[a] || [];
            const bItems = chapters[b] || [];
            const a0 = aItems[0] || null;
            const b0 = bItems[0] || null;

            const primaryOrder = (it) => {
                if (!it) return Number.POSITIVE_INFINITY;
                if (it.source === 'epub') {
                    const n = Number(it.chapter_index);
                    return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY;
                }
                if (it.source === 'pdf') {
                    const n = Number(it.page_index);
                    return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY;
                }
                return Number.POSITIVE_INFINITY;
            };

            const oa = primaryOrder(a0);
            const ob = primaryOrder(b0);
            if (oa !== ob) return oa - ob;

            // Within the same EPUB spine item, try to order by chapter token (VI=6, etc)
            // so NOTES TO CHAPTER VI/VII/VIII/IX sort correctly even if the file is jumbled.
            if (a0 && b0 && a0.source === 'epub' && b0.source === 'epub') {
                const aTok = tryExtractChapterToken(a0.chapter_label) ?? tryExtractChapterToken(a0.chapter_group) ?? tryExtractChapterToken(a);
                const bTok = tryExtractChapterToken(b0.chapter_label) ?? tryExtractChapterToken(b0.chapter_group) ?? tryExtractChapterToken(b);
                if (aTok != null && bTok != null && aTok !== bTok) return aTok - bTok;
                if (aTok != null && bTok == null) return -1;
                if (aTok == null && bTok != null) return 1;

                const ao = minGroupOrder(aItems);
                const bo = minGroupOrder(bItems);
                if (ao !== bo) return ao - bo;
            }

            return a.localeCompare(b, undefined, { numeric: true });
        });

        chapterKeys.forEach(chKey => {
            const tryMarkerInt = (m) => {
                const t = String(m || '').trim();
                if (/^\d{1,3}$/.test(t)) {
                    const n = Number(t);
                    return Number.isFinite(n) ? n : null;
                }
                return null;
            };

            const items = chapters[chKey].slice();

            items.sort((a, b) => {
                // Primary: order by where the anchor occurs in the running text.
                // Do NOT sort by numeric marker first: some EPUBs have out-of-order numbering,
                // and we want the list to match reading order.
                const ao = getOrderValue(a);
                const bo = getOrderValue(b);
                if (ao !== bo) return ao - bo;

                // Tie-breakers: marker value (numeric if possible), then marker string.
                const am = tryMarkerInt(a && a.marker);
                const bm = tryMarkerInt(b && b.marker);
                if (am != null && bm != null && am !== bm) return am - bm;
                const as = String((a && a.marker) || '');
                const bs = String((b && b.marker) || '');
                return as.localeCompare(bs, undefined, { numeric: true });
            });
            const pairedCount = items.filter(x => !!x.suggested_definition).length;
            const reviewCount = items.length - pairedCount;

            const header = document.createElement('div');
            header.className = 'footnote-group';
            header.innerHTML = `
                <div class="group-header">
                    <h3>${getBestChapterLabel(items)} (${pairedCount} paired, ${reviewCount} needs review)</h3>
                </div>
            `;
            container.appendChild(header);

            items.forEach((item, index) => {
                console.log(`Processing item ${index}:`, item);
                const card = document.createElement('div');
                card.className = 'footnote-card';

                const definitionText = item.suggested_definition || "No definition found nearby";
                const renderedContext = formatMultilineText(`...${item.context || ''}...`);
                const renderedDefinition = formatMultilineText(definitionText);
                const score = typeof item.confidence_score === 'number' ? item.confidence_score : 0;
                const isPaired = !!item.suggested_definition;
                const methodMap = {
                    id_link: 'ID Link',
                    marker_unique: 'Marker',
                    marker_order: 'Order',
                    marker_repeat_first: 'Marker (Repeated)',
                    marker_repeat_reuse: 'Marker (Repeated)',
                    marker_repeat_unpaired: 'Repeated (Unpaired)',
                    marker_order_low: 'Order (Low)',
                    ai: 'AI',
                    none: 'Unpaired'
                };
                const methodLabel = methodMap[item.match_method] || item.match_method || 'Unpaired';
                const badgeText = isPaired
                    ? `Paired &bull; ${methodLabel} &bull; ${score.toFixed(2)}`
                    : `Needs review &bull; ${methodLabel} &bull; ${score.toFixed(2)}`;
                const confidenceClass = (
                    isPaired && (
                        score >= 0.75
                        || item.match_method === 'global_unique'
                        || item.match_method === 'recovered_anchor'
                    )
                ) ? "verified-badge" : "warning-badge";

                const hrefLine = item.href ? `<div class="badge" style="margin-top: 6px;">Link: ${escapeHtml(item.href)}</div>` : '';

                card.innerHTML = `
                    <div class="pair-view" style="display: flex; gap: 15px; align-items: flex-start;">
                        <div style="flex: 1;">
                            <span class="badge">STORY CONTEXT</span>
                            <p class="context-text">${renderedContext}</p>
                            <strong>Marker: ${escapeHtml(item.marker)}</strong>
                            ${hrefLine}
                        </div>

                        <div style="align-self: center; font-size: 1.5em; color: #555;">&rarr;</div>

                        <div style="flex: 1; border-left: 1px solid #444; padding-left: 15px;">
                            <span class="badge ${confidenceClass}">${badgeText}</span>
                            <p class="context-text" style="color: #64b5f6;">${renderedDefinition}</p>
                        </div>
                    </div>
                    
                    <div class="card-actions" style="margin-top: 15px; text-align: right;">
                        <button class="ignore-btn">Ignore</button>
                        <button class="confirm-btn" style="background: #3d5afe; color: white;">Confirm Link</button>
                    </div>
                `;
                container.appendChild(card);
                console.log(`Card ${index} created`);
            });
        });

        console.log("Total chapters in container:", chapterKeys.length);

        // Add event listeners for individual Confirm buttons
        document.querySelectorAll('.confirm-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const card = this.closest('.footnote-card');
                if (card) {
                    card.style.opacity = '0.3';
                    card.style.pointerEvents = 'none';
                }
            });
        });

        // Add event listeners for Ignore buttons
        document.querySelectorAll('.ignore-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const card = this.closest('.footnote-card');
                if (card) {
                    card.style.opacity = '0.3';
                    card.style.pointerEvents = 'none';
                }
            });
        });
    }

    function renderTextDisplay(fullText, wordTimestamps, paragraphTimestamps) {
        if (!textDisplayEl || !fullText) return;
        textDisplayEl.style.display = 'block';
        searchBarEl.style.display = 'flex';
        clearSearchHighlights();
        searchInputEl.value = '';

        const hasWords = wordTimestamps && wordTimestamps.length > 0;
        const hasParas = paragraphTimestamps && paragraphTimestamps.length > 0;

        if (hasWords && hasParas) {
            let html = '';
            let wi = 0;
            for (const pt of paragraphTimestamps) {
                html += `<div class="para-block" data-start="${pt.start_ms}" data-end="${pt.end_ms}">`;
                while (wi < wordTimestamps.length && wordTimestamps[wi].start_ms >= pt.start_ms && wordTimestamps[wi].end_ms <= pt.end_ms) {
                    const wt = wordTimestamps[wi];
                    html += `<span class="word-span" data-start="${wt.start_ms}" data-end="${wt.end_ms}">${escapeHtml(wt.word)}</span>`;
                    wi++;
                }
                html += '</div>';
            }
            textDisplayEl.innerHTML = html;
        } else if (hasParas) {
            let html = '';
            for (const pt of paragraphTimestamps) {
                html += `<span data-start="${pt.start_ms}" data-end="${pt.end_ms}" class="para-span">${escapeHtml(pt.text)}</span>`;
            }
            textDisplayEl.innerHTML = html;
        } else if (hasWords) {
            let html = '';
            for (const wt of wordTimestamps) {
                html += `<span class="word-span" data-start="${wt.start_ms}" data-end="${wt.end_ms}">${escapeHtml(wt.word)}</span>`;
            }
            textDisplayEl.innerHTML = html;
        } else {
            textDisplayEl.textContent = fullText;
        }
    }

    function getTtsAudio() {
        return document.getElementById('ttsAudio');
    }

    function saveReadingPosition() {
        if (!currentFilePath) return;
        localStorage.setItem('starlistener_reading_' + currentFilePath, readingPositionMs);
    }

    function applyReadMarks() {
        for (let i = 0; i < allSpans.length; i++) {
            allSpans[i].classList.remove('read');
        }
        for (let i = 0; i < allSpans.length; i++) {
            if (parseFloat(allSpans[i].dataset.end) <= readingPositionMs) {
                allSpans[i].classList.add('read');
            } else {
                break;
            }
        }
        const rb = document.getElementById('resumeBtn');
        if (rb) rb.innerHTML = readingPositionMs > 0 ? 'Resume<br>Reading' : 'Begin<br>Reading';
        saveReadingPosition();
    }

    function startHighlighting(audio, timestamps) {
        if (highlightInterval) {
            audio.removeEventListener('timeupdate', highlightInterval);
        }

        if (currentHighlightEl) {
            currentHighlightEl.classList.remove('highlighted');
            currentHighlightEl = null;
        }

        justSeeked = false;
        isResumePlayback = false;

        allSpans.forEach(function(s) { s.classList.remove('read'); });

        const wordSpans = textDisplayEl.querySelectorAll('.word-span');
        const paraBlocks = textDisplayEl.querySelectorAll('.para-block');
        allSpans = wordSpans.length > 0
            ? [...wordSpans]
            : paraBlocks.length > 0
                ? [...paraBlocks]
                : [...textDisplayEl.querySelectorAll('span[data-start]')];

        function findSpanAt(ms) {
            let lo = 0;
            let hi = allSpans.length - 1;
            while (lo <= hi) {
                const mid = (lo + hi) >> 1;
                const start = parseFloat(allSpans[mid].dataset.start);
                const end = parseFloat(allSpans[mid].dataset.end);
                if (ms >= start && ms < end) return mid;
                if (ms < start) hi = mid - 1;
                else lo = mid + 1;
            }
            return -1;
        }

        function update() {
            const ms = audio.currentTime * 1000;

            const idx = findSpanAt(ms);
            if (idx !== -1) {
                const el = allSpans[idx];
                if (el !== currentHighlightEl) {
                    if (currentHighlightEl) currentHighlightEl.classList.remove('highlighted');
                    currentHighlightEl = el;
                    el.classList.add('highlighted');
                }
            } else if (currentHighlightEl) {
                currentHighlightEl.classList.remove('highlighted');
                currentHighlightEl = null;
            }

            if (!audio.paused && ms > readingPositionMs) {
                if (justSeeked) {
                    justSeeked = false;
                } else if (isResumePlayback) {
                    readingPositionMs = ms;
                    applyReadMarks();
                }
            }
        }

        highlightInterval = update;
        audio.addEventListener('timeupdate', update);

        if (readingPositionMs > 0) {
            applyReadMarks();
        }

        const transportBar = document.getElementById('transportBar');
        if (transportBar) transportBar.style.display = 'flex';

        const playBtn = document.getElementById('playPauseBtn');
        if (playBtn) playBtn.textContent = audio.paused ? '\u25B6 Play' : '\u23F8 Pause';

        audio.addEventListener('play', function() {
            const btn = document.getElementById('playPauseBtn');
            if (btn) btn.textContent = '\u23F8 Pause';
            if (currentHighlightEl) {
                currentHighlightEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }
        });
        audio.addEventListener('pause', function() {
            const btn = document.getElementById('playPauseBtn');
            if (btn) btn.textContent = '\u25B6 Play';
            isResumePlayback = false;
        });
        audio.addEventListener('ended', function() {
            const btn = document.getElementById('playPauseBtn');
            if (btn) btn.textContent = '\u25B6 Play';
            isResumePlayback = false;
        });

        audio.addEventListener('seeking', function() {
            justSeeked = true;
        });
        audio.addEventListener('seeked', function() {
            justSeeked = true;
            if (currentHighlightEl) {
                currentHighlightEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }
        });

        const rb = document.getElementById('resumeBtn');
        if (rb) rb.innerHTML = readingPositionMs > 0 ? 'Resume<br>Reading' : 'Begin<br>Reading';
    }

    textDisplayEl.addEventListener('click', function(e) {
        const audio = getTtsAudio();
        if (!audio) return;
        const el = e.target.closest('.word-span') || e.target.closest('.para-block') || e.target.closest('[data-start]');
        if (!el) return;
        audio.currentTime = parseFloat(el.dataset.start) / 1000;
    });

    const playPauseBtn = document.getElementById('playPauseBtn');
    playPauseBtn.addEventListener('click', function() {
        const a = getTtsAudio();
        if (!a) return;
        if (a.paused) a.play();
        else a.pause();
    });

    const resumeBtn = document.getElementById('resumeBtn');
    resumeBtn.addEventListener('click', function() {
        const a = getTtsAudio();
        if (!a) return;
        isResumePlayback = true;
        a.currentTime = readingPositionMs / 1000;
        a.play();
    });

    const resumeHereBtn = document.getElementById('resumeHereBtn');
    resumeHereBtn.addEventListener('click', function() {
        const a = getTtsAudio();
        if (!a) return;
        isResumePlayback = true;
        readingPositionMs = Math.max(a.currentTime * 1000, readingPositionMs);
        applyReadMarks();
        a.play();
    });

    function skip(seconds) {
        const a = getTtsAudio();
        if (!a) return;
        a.currentTime = Math.max(0, Math.min(a.duration || 0, a.currentTime + seconds));
    }

    document.getElementById('skipBack5m').addEventListener('click', function() { skip(-300); });
    document.getElementById('skipBack1m').addEventListener('click', function() { skip(-60); });
    document.getElementById('skipBack10s').addEventListener('click', function() { skip(-10); });
    document.getElementById('skipFwd10s').addEventListener('click', function() { skip(10); });
    document.getElementById('skipFwd1m').addEventListener('click', function() { skip(60); });
    document.getElementById('skipFwd5m').addEventListener('click', function() { skip(300); });

    function clearSearchHighlights() {
        if (!textDisplayEl) return;
        textDisplayEl.querySelectorAll('mark.search-match').forEach(mark => {
            mark.replaceWith(mark.textContent);
        });
        searchMatches = [];
        currentMatchIdx = -1;
        updateSearchUI();
    }

    function updateSearchUI() {
        const total = searchMatches.length;
        if (total > 0) {
            searchStatusEl.textContent = `${currentMatchIdx + 1}/${total}`;
            findPrevBtn.disabled = false;
            findNextBtn.disabled = false;
        } else {
            searchStatusEl.textContent = '';
            findPrevBtn.disabled = true;
            findNextBtn.disabled = true;
        }
    }

    function goToMatch(idx) {
        searchMatches.forEach(m => m.classList.remove('current'));
        if (idx >= 0 && idx < searchMatches.length) {
            searchMatches[idx].classList.add('current');
            searchMatches[idx].scrollIntoView({ block: 'center', behavior: 'smooth' });
            currentMatchIdx = idx;
        }
        updateSearchUI();
    }

    function findNext() {
        if (searchMatches.length === 0) return;
        currentMatchIdx = (currentMatchIdx + 1) % searchMatches.length;
        goToMatch(currentMatchIdx);
    }

    function findPrev() {
        if (searchMatches.length === 0) return;
        currentMatchIdx = (currentMatchIdx - 1 + searchMatches.length) % searchMatches.length;
        goToMatch(currentMatchIdx);
    }

    function performSearch(query) {
        if (!textDisplayEl) return;
        clearSearchHighlights();
        if (!query || !query.trim()) return;

        const rawQuery = query.trim();
        const escapedRegexStr = rawQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(escapedRegexStr, 'gi');

        const walker = document.createTreeWalker(textDisplayEl, NodeFilter.SHOW_TEXT);
        const segments = [];
        let cumPos = 0;
        let wn;
        while ((wn = walker.nextNode())) {
            const len = wn.textContent.length;
            segments.push({ node: wn, start: cumPos, end: cumPos + len });
            cumPos += len;
        }

        if (segments.length === 0) return;

        const fullText = segments.map(s => s.node.textContent).join('');

        regex.lastIndex = 0;
        let match;
        const matchRanges = [];
        while ((match = regex.exec(fullText)) !== null) {
            matchRanges.push({ start: match.index, end: match.index + match[0].length });
            if (match[0].length === 0) regex.lastIndex++;
        }

        if (matchRanges.length === 0) {
            updateSearchUI();
            return;
        }

        for (let i = segments.length - 1; i >= 0; i--) {
            const seg = segments[i];

            const boundaries = new Set([seg.start, seg.end]);
            let hasOverlap = false;
            for (const mr of matchRanges) {
                if (mr.start < seg.end && mr.end > seg.start) {
                    hasOverlap = true;
                    boundaries.add(Math.max(mr.start, seg.start));
                    boundaries.add(Math.min(mr.end, seg.end));
                }
            }

            if (!hasOverlap) continue;

            const sortedBounds = [...boundaries].sort((a, b) => a - b);
            const frag = document.createDocumentFragment();
            const text = seg.node.textContent;
            const segStart = seg.start;

            for (let j = 0; j < sortedBounds.length - 1; j++) {
                const a = sortedBounds[j];
                const b = sortedBounds[j + 1];
                const localA = a - segStart;
                const localB = b - segStart;

                let inMatch = false;
                for (const mr of matchRanges) {
                    if (a >= mr.start && b <= mr.end) {
                        inMatch = true;
                        break;
                    }
                }

                if (inMatch) {
                    const mk = document.createElement('mark');
                    mk.className = 'search-match';
                    mk.textContent = text.slice(localA, localB);
                    frag.appendChild(mk);
                } else {
                    frag.appendChild(document.createTextNode(text.slice(localA, localB)));
                }
            }

            seg.node.parentNode.replaceChild(frag, seg.node);
        }

        searchMatches = [...textDisplayEl.querySelectorAll('mark.search-match')];
        if (searchMatches.length > 0) {
            currentMatchIdx = 0;
            goToMatch(0);
        }
        updateSearchUI();
    }

    searchBarEl.style.display = 'none';

    findBtn.addEventListener('click', () => performSearch(searchInputEl.value));
    clearSearchBtn.addEventListener('click', () => {
        clearSearchHighlights();
        searchInputEl.value = '';
    });
    findNextBtn.addEventListener('click', findNext);
    findPrevBtn.addEventListener('click', findPrev);
    searchInputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            performSearch(searchInputEl.value);
        } else if (e.key === 'Escape') {
            clearSearchHighlights();
            searchInputEl.value = '';
        }
    });

    function getBookDataPaths(bookPath) {
        const fname = bookPath.split(/[/\\]/).pop();
        const safeFolder = fname.replace(/\s+/g, '_');
        const stem = fname.replace(/\.[^.]+$/, '').replace(/\s+/g, '_');
        return {
            nmPath: `output\\${safeFolder}\\${stem}_notemarkers.json`,
            tsPath: `output\\${safeFolder}\\${stem}_timestamps.json`,
            mp3Path: `output\\${safeFolder}\\${stem}.mp3`,
        };
    }

    async function loadBookData(bookPath) {
        const p = getBookDataPaths(bookPath);

        try {
            const scanResults = await window.electronAPI.readJsonFile(p.nmPath);
            if (scanResults && scanResults.length > 0) {
                lastScanResults = scanResults;
                footnoteOptionsEl.style.display = 'flex';
                footnoteModeSelect.disabled = false;
                const fMode = localStorage.getItem('starlistener_footnoteMode');
                if (fMode) footnoteModeSelect.value = fMode;
                showFootnotes(JSON.stringify(scanResults));
            }
        } catch (e) {}

        try {
            const ts = await window.electronAPI.readJsonFile(p.tsPath);
            if (ts && ts.full_text && ts.paragraphs && ts.paragraphs.length > 0) {
                const wordTimestamps = ts.timestamps || [];
                const paraTimestamps = ts.paragraphs || [];
                renderTextDisplay(ts.full_text, wordTimestamps, paraTimestamps);

                const existingAudio = document.getElementById('ttsAudio');
                if (existingAudio) existingAudio.remove();

                const absMp3 = await window.electronAPI.resolvePath(p.mp3Path);
                const audio = document.createElement('audio');
                audio.id = 'ttsAudio';
                audio.controls = true;
                audio.src = `file:///${absMp3.replace(/\\/g, '/')}`;
                audio.style.cssText = 'margin-top: 12px; width: 100%;';
                document.getElementById('audioContainer').appendChild(audio);

                startHighlighting(audio, paraTimestamps);
                output.innerText = 'Loaded session';
            }
        } catch (e) {}
    }

    async function restoreSession() {
        try {
            const saved = await window.electronAPI.loadSessionInfo();
            if (!saved || !saved.bookPath) return;

            currentFilePath = saved.bookPath;
            const basename = currentFilePath.split(/[/\\]/).pop();
            currentFileEl.innerText = basename;
            scanBtn.disabled = false;
            ttsBtn.disabled = false;

            const voice = localStorage.getItem('starlistener_mainVoice');
            const footnoteVoice = localStorage.getItem('starlistener_footnoteVoice');
            if (voice) mainVoiceSelect.value = voice;
            if (footnoteVoice) footnoteVoiceSelect.value = footnoteVoice;

            const savedMarker = localStorage.getItem('starlistener_markerProfile');
            if (savedMarker) markerProfileSelect.value = savedMarker;
            const savedSpeed = localStorage.getItem('starlistener_playbackSpeed');
            if (savedSpeed) playbackSpeedSelect.value = savedSpeed;

            readingPositionMs = saved.readingPositionMs || 0;
            const localKey = 'starlistener_reading_' + saved.bookPath;
            const localPos = localStorage.getItem(localKey);
            if (localPos != null) readingPositionMs = Number(localPos);

            await loadBookData(currentFilePath);
        } catch (e) {
            console.error('Session restore failed:', e);
        }
    }

    restoreSession();

});