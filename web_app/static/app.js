/* ══════════════════════════════════════════════════════
   乾坤 AI 科學面相館 — app.js v4.0
   全新算命館玄學風 · 垂直捲動流佈局
   ══════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    /* ─────────────────────────────
       狀態物件
    ───────────────────────────── */
    const state = {
        file:         null,
        activeTab:    'tab-upload',
        provider:     'default',
        webcamStream: null,
        tipTimer:     null,
        stepTimer:    null,
        currentStep:  0,
        lastResult:   null,
    };

    /* ─────────────────────────────
       DOM 快速取用
    ───────────────────────────── */
    const $ = id => document.getElementById(id);

    // 分頁
    const tabBtns     = document.querySelectorAll('.tab-btn');
    const tabPanes    = document.querySelectorAll('.tab-pane');

    // 上傳
    const dropzone    = $('dropzone');
    const fileInput   = $('file-input');
    const uploadPanel = $('upload-panel');
    const previewPanel = $('preview-panel');
    const imgPreview  = $('img-preview');
    const scanOverlay = $('scan-overlay');
    const previewBar  = $('preview-bar');
    const resetBtn    = $('reset-btn');
    const analyzeBtn  = $('analyze-btn');

    // 相機
    const webcamVideo  = $('webcam-video');
    const webcamCanvas = $('webcam-canvas');
    const captureBtn   = $('capture-btn');

    // Config
    const apiKeyInput = $('api-key-input');
    const keyToggle   = $('key-toggle');
    const keyEye      = $('key-eye');
    const apiKeySection = $('api-key-section');
    const devKeyNotice  = $('dev-key-notice');

    // Status
    const cnnDot       = $('cnn-dot');
    const cnnStatusText = $('cnn-status-text');

    // States
    const stateLoading = $('state-loading');
    const stateResult  = $('state-result');

    // Loading
    const tipText = $('tip-text');

    // Result
    const annotatedImg    = $('annotated-img');
    const geoList         = $('geo-list');
    const cnnList         = $('cnn-list');
    const ruleSummary     = $('rule-summary');
    const cnnSummary      = $('cnn-summary');
    const llmReport       = $('llm-report');
    const rsModelTag      = $('rs-model-tag');
    const resultModelTag  = $('result-model-tag');
    const resultTime      = $('result-time');
    const printBtn        = $('print-btn');
    const downloadBtn     = $('download-btn');
    const reanalyzeBtn    = $('reanalyze-btn');

    /* ─────────────────────────────
       載入提示文字
    ───────────────────────────── */
    const TIPS = [
        'MediaPipe Face Mesh 定位 468 個特徵點...',
        '計算三庭五眼幾何黃金比例...',
        '裁切五官 ROI，準備 CNN 輸入...',
        'MobileNetV2 特徵提取中...',
        '深度學習五官分類推理中...',
        '整合分析結果，生成 LLM Prompt...',
        'AI 命理引擎生成個性化批命報告...',
        '報告完成，渲染結果...',
    ];

    /* ─────────────────────────────
       1. 查詢後端狀態
    ───────────────────────────── */
    async function initStatus() {
        try {
            const res = await fetch('/api/status');
            if (!res.ok) throw new Error('failed');
            const data = await res.json();

            if (data.cnn_ready) {
                cnnDot.className = 'sys-dot sys-dot-green status-ready';
                cnnStatusText.textContent = 'CNN 掃描器引擎就緒 ✓';
            } else {
                cnnDot.className = 'sys-dot status-amber';
                cnnStatusText.textContent = 'CNN 掃描器 (模擬中)';
            }

            // 開發者金鑰 → provider-card
            const devKeys = data.dev_keys_available || {};
            document.querySelectorAll('.provider-card').forEach(btn => {
                const p = btn.getAttribute('data-provider');
                if (devKeys[p] && !btn.querySelector('.dev-mark')) {
                    const m = document.createElement('span');
                    m.className = 'dev-mark';
                    m.textContent = '✓';
                    btn.appendChild(m);
                }
            });
        } catch {
            cnnStatusText.textContent = '無法連接後端伺服器';
            cnnDot.style.background = '#ef4444';
        }
    }
    initStatus();

    /* ─────────────────────────────
       2. 分頁切換（上傳 / 相機）
    ───────────────────────────── */
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.toggle('active', b === btn));
            tabPanes.forEach(p => p.classList.toggle('active', p.id === target));
            state.activeTab = target;
            if (target === 'tab-webcam') startWebcam();
            else stopWebcam();
        });
    });

    /* ─────────────────────────────
       3. 拖放上傳
    ───────────────────────────── */
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => { if (e.target.files[0]) pickFile(e.target.files[0]); });
    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
    dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dropzone.classList.remove('over');
        if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
    });

    function pickFile(file) {
        if (!file.type.startsWith('image/')) { alert('請選擇圖片格式（JPG / PNG）'); return; }
        state.file = file;
        const reader = new FileReader();
        reader.onload = e => {
            imgPreview.src = e.target.result;
            showPreview();
        };
        reader.readAsDataURL(file);
    }

    function showPreview() {
        uploadPanel.classList.add('sr-only');
        previewPanel.classList.remove('sr-only');
        // 啟用按鈕
        resetBtn.disabled = false;
        analyzeBtn.disabled = false;
        lucide.createIcons();
    }

    function hidePreview() {
        previewPanel.classList.add('sr-only');
        uploadPanel.classList.remove('sr-only');
        scanOverlay.classList.add('sr-only');
        // 禁用按鈕
        resetBtn.disabled = true;
        analyzeBtn.disabled = true;
        imgPreview.src = '';
        // 恢復上傳分頁
        tabPanes.forEach(p => p.classList.toggle('active', p.id === state.activeTab));
        lucide.createIcons();
    }

    function formatMarkdownBold(text) {
        if (!text) return '';
        let cleaned = text.replace(/\\\*/g, '*');
        return cleaned.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }

    resetBtn.addEventListener('click', () => {
        state.file = null;
        fileInput.value = '';
        state.lastResult = null;
        hidePreview();
        stateResult.classList.add('sr-only');
        if (state.activeTab === 'tab-webcam') startWebcam();
    });

    /* ─────────────────────────────
       4. 相機
    ───────────────────────────── */
    async function startWebcam() {
        try {
            state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            webcamVideo.srcObject = state.webcamStream;
        } catch {
            alert('無法啟動相機，請確認瀏覽器授權或改用「上傳照片」模式。');
            tabBtns[0].click();
        }
    }

    function stopWebcam() {
        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach(t => t.stop());
            state.webcamStream = null;
        }
        webcamVideo.srcObject = null;
    }

    captureBtn.addEventListener('click', () => {
        if (!webcamVideo.srcObject) return;
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext('2d');
        ctx.translate(webcamCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0);
        webcamCanvas.toBlob(blob => {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            stopWebcam();
            pickFile(file);
        }, 'image/jpeg', 0.95);
    });

    /* ─────────────────────────────
       5. LLM Provider 切換
    ───────────────────────────── */
    function updateApiKeyVisibility() {
        const isDefault = (state.provider === 'default');
        if (apiKeySection) {
            if (isDefault) {
                apiKeySection.classList.add('disabled');
            } else {
                apiKeySection.classList.remove('disabled');
            }
        }
        if (apiKeyInput) {
            apiKeyInput.disabled = isDefault;
            apiKeyInput.placeholder = isDefault
                ? '預設引擎不需要 Key'
                : {
                    claude: 'sk-ant-... (Anthropic)',
                    openai: 'sk-... (OpenAI)',
                    gemini: 'AIza... (Google)'
                  }[state.provider] || '貼上您的 API Key';
        }
    }
    updateApiKeyVisibility(); // 初始化

    document.querySelectorAll('.provider-card').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.provider-card').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.provider = btn.getAttribute('data-provider');
            updateApiKeyVisibility();
        });
    });

    /* ─────────────────────────────
       6. Key 顯示切換
    ───────────────────────────── */
    keyToggle.addEventListener('click', () => {
        const show = apiKeyInput.type === 'password';
        apiKeyInput.type = show ? 'text' : 'password';
        keyEye.setAttribute('data-lucide', show ? 'eye-off' : 'eye');
        lucide.createIcons({ nodes: [keyEye] });
    });

    /* ─────────────────────────────
       8. 解讀風格
    ───────────────────────────── */
    document.querySelectorAll('.style-chip input').forEach(radio => {
        radio.addEventListener('change', () => {
            document.querySelectorAll('.style-chip').forEach(c => {
                c.classList.toggle('active', c.querySelector('input') === radio);
            });
        });
    });

    /* ─────────────────────────────
       9. 開始分析
    ───────────────────────────── */
    analyzeBtn.addEventListener('click', () => {
        if (!state.file) return;
        runAnalysis();
    });

    reanalyzeBtn.addEventListener('click', () => {
        // 收起結果區
        stateResult.classList.add('sr-only');
        stateLoading.classList.add('sr-only');
        scanOverlay.classList.add('sr-only');
        // 啟用按鈕狀態
        analyzeBtn.disabled = false;
        resetBtn.disabled = false;
        // 滚回上傳區（保留已上傳的圖片預覽，可直接修改解讀項目與重新批命）
        $('analyzer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    async function runAnalysis() {
        scanOverlay.classList.remove('sr-only');
        analyzeBtn.disabled = true;
        resetBtn.disabled = true;

        // 顯示 loading
        stateLoading.classList.remove('sr-only');
        stateResult.classList.add('sr-only');
        startLoadingAnim();

        // 滾動到 loading
        stateLoading.scrollIntoView({ behavior: 'smooth', block: 'center' });

        const style   = document.querySelector('input[name="style"]:checked')?.value || 'traditional';
        const userKey = apiKeyInput.value.trim();

        const form = new FormData();
        form.append('file',     state.file);
        form.append('style',    style);
        const actualProvider = (state.provider === 'default') ? 'gemini' : state.provider;
        form.append('provider', actualProvider);
        if (userKey && state.provider !== 'default') form.append('api_key', userKey);

        try {
            const res = await fetch('/api/analyze', { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.message || err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            stopLoadingAnim();
            stateLoading.classList.add('sr-only');
            renderResults(data);
            stateResult.classList.remove('sr-only');
            // 啟用按鈕狀態，讓用戶可重新批命或更換設定
            analyzeBtn.disabled = false;
            resetBtn.disabled = false;
            setTimeout(() => stateResult.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
        } catch (err) {
            stopLoadingAnim();
            stateLoading.classList.add('sr-only');
            scanOverlay.classList.add('sr-only');
            analyzeBtn.disabled = false;
            resetBtn.disabled = false;
            alert(`❌ 分析失敗：${err.message}`);
        }
    }

    /* ─────────────────────────────
       10. 載入動畫
    ───────────────────────────── */
    function startLoadingAnim() {
        state.currentStep = 0;
        [0,1,2,3].forEach(i => setStep(i, ''));
        setStep(0, 'active');
        tipText.textContent = TIPS[0];

        let idx = 0;
        state.tipTimer = setInterval(() => {
            idx = (idx + 1) % TIPS.length;
            tipText.style.opacity = '0';
            setTimeout(() => { tipText.textContent = TIPS[idx]; tipText.style.opacity = '1'; }, 300);
        }, 3500);

        state.stepTimer = setInterval(() => {
            if (state.currentStep < 3) {
                setStep(state.currentStep, 'done');
                state.currentStep++;
                setStep(state.currentStep, 'active');
            }
        }, 7000);
    }

    function stopLoadingAnim() {
        clearInterval(state.tipTimer);
        clearInterval(state.stepTimer);
        [0,1,2,3].forEach(i => setStep(i, 'done'));
    }

    function setStep(idx, status) {
        const el = $(`si-${idx}`);
        if (!el) return;
        el.classList.remove('active', 'done');
        if (status) el.classList.add(status);
    }

    /* ─────────────────────────────
       11. 渲染結果
    ───────────────────────────── */
    function renderResults(data) {
        state.lastResult = data;
        // 時間戳 & 模型名
        const now = new Date();
        resultTime.textContent = now.toLocaleTimeString('zh-TW');
        const modelName = data.llm_model_name || '—';
        resultModelTag.textContent = modelName;
        rsModelTag.textContent = modelName;

        // CNN 狀態
        if (data.is_real_cnn) {
            cnnDot.className = 'sys-dot sys-dot-green';
            cnnStatusText.textContent = 'CNN 掃描器引擎就緒 ✓';
        } else {
            cnnDot.className = 'sys-dot status-amber';
            cnnStatusText.textContent = 'CNN 掃描器 (模擬中)';
        }

        // 標註圖
        annotatedImg.src = data.annotated_image;

        // ─ 路線 A：幾何特徵 ─
        const f = data.features || {};
        const avgEye = ((f.left_eye_ratio || 0) + (f.right_eye_ratio || 0)) / 2;
        const geoData = [
            { name: '臉部寬高比',  val: f.face_ratio,             barW: (f.face_ratio || 0) * 100,         note: noteGeo('face',  f.face_ratio) },
            { name: '三庭均衡度',  val: f.san_ting_balance,       barW: (f.san_ting_balance || 0) * 100,    note: noteGeo('sting', f.san_ting_balance) },
            { name: '眼部開闊度',  val: avgEye,                   barW: avgEye * 100,                        note: noteGeo('eye',   avgEye) },
            { name: '鼻寬比例',    val: f.nose_width_ratio,       barW: (f.nose_width_ratio || 0) * 200,    note: noteGeo('nose',  f.nose_width_ratio) },
            { name: '嘴寬比例',    val: f.mouth_width_ratio,      barW: (f.mouth_width_ratio || 0) * 180,   note: noteGeo('mouth', f.mouth_width_ratio) },
            { name: '印堂寬度比',  val: f.glabella_width_ratio,   barW: (f.glabella_width_ratio || 0) * 300, note: noteGeo('glabella', f.glabella_width_ratio) },
            { name: '眉壓眼指數',  val: f.brow_eye_distance,      barW: (f.brow_eye_distance || 0) * 600,   note: noteGeo('broweye', f.brow_eye_distance) },
            { name: '人中長度比',  val: f.philtrum_ratio,         barW: (f.philtrum_ratio || 0) * 1200,   note: noteGeo('philtrum', f.philtrum_ratio) }
        ];

        geoList.innerHTML = geoData.map(({ name, val, barW, note }) => {
            const w = Math.min(Math.max(barW || 0, 2), 100).toFixed(0);
            return `<li class="metric-item">
                <div class="metric-row">
                    <span class="metric-name">${name}</span>
                    <span class="metric-val">${(val||0).toFixed(3)} &middot; ${note}</span>
                </div>
                <div class="bar-track"><div class="bar-fill-a" style="width:0%" data-w="${w}%"></div></div>
            </li>`;
        }).join('');

        if (data.rule_report && typeof data.rule_report === 'object') {
            let ruleHtml = `<strong class="title-label">【綜合臉型格局】</strong> ${data.rule_report.face_type || '—'}<br><br>`;
            let summaryText = formatMarkdownBold(data.rule_report.summary || '—');
            ruleHtml += `${summaryText.replace(/\n/g, '<br>')}`;
            if (data.rule_report.lucky_hint) {
                let hintText = formatMarkdownBold(data.rule_report.lucky_hint);
                ruleHtml += `<br><br>🍀 <strong class="title-label">開運智慧：</strong>${hintText}`;
            }
            ruleSummary.innerHTML = ruleHtml;
        } else {
            let ruleText = formatMarkdownBold(data.rule_report || '—');
            ruleSummary.innerHTML = ruleText.replace(/\n/g, '<br>');
        }

        // ─ 路線 B：CNN ─
        const iconMap = { nose:'👃', eye:'👁️', brow:'〰️', mouth:'👄', face:'🔷' };
        const nameMap = { nose:'鼻型', eye:'眼型', brow:'眉型', mouth:'嘴型', face:'臉型' };

        cnnList.innerHTML = Object.entries(data.cnn_preds || {}).map(([k, pred]) => {
            const pct = Math.round((pred.confidence || 0) * 100);
            const ensembleTag = pred.best_exp && pred.val_acc ?
                `<div class="cnn-source-tag">🎯 AI 引擎: ${pred.best_exp} (精度: ${pred.val_acc})</div>` : '';
            return `<div class="cnn-item">
                <div class="cnn-row">
                    <span class="cnn-name">${iconMap[k]||''} ${nameMap[k]||k}：<strong>${pred.label}</strong></span>
                    <span class="cnn-val">${pct}%</span>
                </div>
                <div class="bar-track"><div class="bar-fill-b" style="width:0%" data-w="${pct}%"></div></div>
                ${ensembleTag}
            </div>`;
        }).join('');

        let cnnSummaryText = formatMarkdownBold((data.cnn_report || {}).summary || '—');
        cnnSummary.innerHTML = cnnSummaryText.replace(/\n/g, '<br>');

        // ─ 五宮評分 ─
        const palaceContainer = $('palace-scores-container');
        const palaceGrid = $('palace-grid');
        if (data.palace_scores && Object.keys(data.palace_scores).length > 0) {
            palaceContainer.style.display = 'block';
            palaceGrid.innerHTML = Object.entries(data.palace_scores).map(([name, score]) => {
                return `<div class="palace-card">
                    <div class="palace-name">${name}</div>
                    <div class="palace-score-num">${score}</div>
                    <div class="palace-progress">
                        <div class="palace-progress-fill" style="width: 0%" data-w="${score}%"></div>
                    </div>
                </div>`;
            }).join('');
        } else {
            palaceContainer.style.display = 'none';
        }

        // 觸發進度條動畫
        requestAnimationFrame(() => requestAnimationFrame(() => {
            document.querySelectorAll('.bar-fill-a, .bar-fill-b, .palace-progress-fill').forEach(el => {
                el.style.width = el.dataset.w || '0%';
            });
        }));

        // ─ LLM 報告 ─
        let reportText = data.llm_report || '（LLM 未回傳內容）';
        // 移去可能被 LLM 逸出轉義的反斜線，例如將 \*\* 轉回 **
        reportText = reportText.replace(/\\\*/g, '*');
        llmReport.innerHTML = marked.parse(reportText);
        lucide.createIcons();
    }

    /* ─────────────────────────────
       12. 輔助函數
    ───────────────────────────── */
    function noteGeo(type, val) {
        if (!val) return '—';
        const notes = {
            face:  val > .8 ? '圓潤' : val < .65 ? '細長' : '標準',
            sting: val > .85 ? '均衡' : val < .7 ? '偏差' : '略差',
            eye:   val > .32 ? '大眼' : val < .2 ? '細長' : '標準',
            nose:  val > .35 ? '寬闊' : val < .2 ? '纖細' : '標準',
            mouth: val > .42 ? '寬大' : val < .3 ? '小巧' : '標準',
            glabella: val > .18 ? '開闊' : val < .12 ? '偏窄' : '適中',
            broweye: val > .08 ? '寬廣' : val < .055 ? '壓眼' : '適中',
            philtrum: val > .04 ? '深長' : val < .028 ? '偏短' : '適中'
        };
        return notes[type] || '—';
    }

    /* ─────────────────────────────
       13. 列印與下載
    ───────────────────────────── */
    printBtn.addEventListener('click', () => window.print());

    downloadBtn.addEventListener('click', () => {
        const data = state.lastResult;
        if (!data) {
            alert('尚無分析資料可供下載');
            return;
        }

        const now = new Date();
        const timeStr = now.toLocaleString('zh-TW');
        const modelName = data.llm_model_name || 'Gemini 3.5 Flash';

        let txt = `🔮 乾坤 AI 科學面相館 — 完整面相鑑定報告\n`;
        txt += `==================================================\n`;
        txt += `鑑定時間：${timeStr}\n`;
        txt += `分析引擎：${modelName}\n`;
        txt += `==================================================\n\n`;

        // Palace Scores
        if (data.palace_scores && Object.keys(data.palace_scores).length > 0) {
            txt += `【整體五宮命理評分】\n`;
            txt += `--------------------------------------------------\n`;
            Object.entries(data.palace_scores).forEach(([palace, score]) => {
                txt += ` - ${palace}宮：${score} 分\n`;
            });
            txt += `\n`;
        }

        // Route A
        txt += `==================================================\n`;
        txt += `【路線 A：幾何規則分析 (三庭五眼 · 黃金比例)】\n`;
        txt += `--------------------------------------------------\n`;
        if (data.rule_report && typeof data.rule_report === 'object') {
            txt += `臉型格局：${data.rule_report.face_type || '—'}\n\n`;
            txt += `幾何量測明細與格局解析：\n`;
            txt += `${data.rule_report.summary || '—'}\n`;
        } else {
            txt += `${data.rule_report || '—'}\n`;
        }
        txt += `\n`;

        // Route B
        txt += `==================================================\n`;
        txt += `【路線 B：CNN 深度學習 (五官局部圖像分類)】\n`;
        txt += `--------------------------------------------------\n`;
        if (data.cnn_report && typeof data.cnn_report === 'object') {
            txt += `${data.cnn_report.summary || '—'}\n`;
        } else {
            txt += `${data.cnn_report || '—'}\n`;
        }
        txt += `\n`;

        // Route C
        txt += `==================================================\n`;
        txt += `【路線 C：AI 命理鑑定書 (多模態大師批命)】\n`;
        txt += `==================================================\n`;
        txt += `${data.llm_report || '（尚無報告內容）'}\n\n`;

        txt += `==================================================\n`;
        if (data.rule_report && data.rule_report.lucky_hint) {
            txt += `🍀【大師開運智慧】\n`;
            txt += `--------------------------------------------------\n`;
            txt += `${data.rule_report.lucky_hint}\n\n`;
            txt += `==================================================\n`;
        }
        txt += `報告完結。天機啟迪，運勢亨通！\n`;

        // Trigger client-side file download
        const blob = new Blob([txt], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `乾坤面相鑑定報告_${now.getFullYear()}${(now.getMonth()+1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    /* ─────────────────────────────
       14. 平滑錨點
    ───────────────────────────── */
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            const target = document.querySelector(a.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
});
