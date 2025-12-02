// Estado global de la aplicación
const appState = {
    loadedResults: [],
    currentResult: null,
    currentMethod: 'cone',
    dataSource: null,  // 'github' o 'local'
    githubAbortController: null  // Para cancelar carga de GitHub
};

// Colores por dimensión (paleta científica sobria)
const DIMENSION_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6'];
const DIM_LABELS = ['H₀', 'H₁', 'H₂', 'H₃', 'H₄'];

// Funciones para localStorage
function saveStateToLocalStorage() {
    try {
        const state = {
            loadedResults: appState.loadedResults,
            currentMethod: appState.currentMethod,
            dataSource: appState.dataSource,
            selectedExperiment: document.getElementById('experimentSelect')?.value,
            selectedN: document.getElementById('nSelect')?.value,
            selectedEps: document.getElementById('epsSelect')?.value
        };
        localStorage.setItem('persistentCostViewerState', JSON.stringify(state));
    } catch (e) {
        console.warn('No se pudo guardar el estado:', e);
    }
}

function loadStateFromLocalStorage() {
    console.log('[STORAGE] Intentando cargar estado desde localStorage...');
    try {
        const saved = localStorage.getItem('persistentCostViewerState');
        if (!saved) {
            console.log('[STORAGE] No hay estado guardado');
            return false;
        }
        
        const state = JSON.parse(saved);
        console.log('[STORAGE] Estado encontrado - dataSource:', state.dataSource, 'resultados:', state.loadedResults?.length);
        
        if (state.loadedResults && state.loadedResults.length > 0) {
            appState.loadedResults = state.loadedResults;
            appState.currentMethod = state.currentMethod || 'cone';
            appState.dataSource = state.dataSource || 'github';
            console.log('[STORAGE] Estado cargado - Método:', appState.currentMethod, 'Source:', appState.dataSource);
            
            updateDataSourceIndicator();
            populateSelectors();
            document.getElementById('resultsSelector').style.display = 'block';
            
            // Restaurar selecciones
            if (state.selectedExperiment) {
                console.log('[STORAGE] Restaurando selección - Exp:', state.selectedExperiment, 'N:', state.selectedN, 'Eps:', state.selectedEps);
                document.getElementById('experimentSelect').value = state.selectedExperiment;
                updateNSelect();
                
                if (state.selectedN) {
                    document.getElementById('nSelect').value = state.selectedN;
                    updateEpsSelect();
                    
                    if (state.selectedEps !== undefined) {
                        setTimeout(() => {
                            const epsSelect = document.getElementById('epsSelect');
                            epsSelect.value = state.selectedEps;
                            displaySelectedResult();
                        }, 100);
                    }
                }
            } else {
                // Si no hay selección guardada, seleccionar el primer experimento
                console.log('[STORAGE] No hay selección guardada, seleccionando primer experimento');
                const experiments = [...new Set(appState.loadedResults.map(r => r.experiment_name))].sort();
                if (experiments.length > 0) {
                    document.getElementById('experimentSelect').value = experiments[0];
                    updateNSelect();
                }
            }
            
            return true;
        }
    } catch (e) {
        console.warn('[STORAGE] Error cargando estado:', e);
    }
    return false;
}

function updateDataSourceIndicator() {
    console.log('[UI] updateDataSourceIndicator - source:', appState.dataSource, 'count:', appState.loadedResults.length);
    const indicator = document.getElementById('dataSourceIndicator');
    if (!indicator) return;
    
    const count = appState.loadedResults.length;
    
    if (appState.dataSource === 'local') {
        indicator.innerHTML = `<span class="source-badge source-local">Local<span class="source-count">(${count} archivos)</span></span>`;
    } else if (appState.dataSource === 'github') {
        indicator.innerHTML = `<span class="source-badge source-github">GitHub<span class="source-count">(${count} archivos)</span></span>`;
    } else {
        indicator.innerHTML = `<span class="source-badge source-loading">Cargando...</span>`;
    }
}

function showCacheIndicator() {
    const indicator = document.getElementById('cacheIndicator');
    if (indicator) {
        indicator.style.display = 'inline-block';
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 3000);
    }
}

function clearCache() {
    if (confirm('¿Estás seguro de que quieres limpiar los datos guardados? Esto recargará los datos desde GitHub.')) {
        try {
            localStorage.removeItem('persistentCostViewerState');
            location.reload();
        } catch (e) {
            console.warn('No se pudo limpiar el caché:', e);
        }
    }
}

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    console.log('[INIT] DOMContentLoaded - Iniciando aplicación');
    const folderInput = document.getElementById('folderInput');
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    const showPersistenceToggle = document.getElementById('showPersistenceMetrics');
    
    folderInput.addEventListener('change', handleFolderSelect);
    clearCacheBtn.addEventListener('click', clearCache);
    
    const experimentSelect = document.getElementById('experimentSelect');
    const nSelect = document.getElementById('nSelect');
    const epsSelect = document.getElementById('epsSelect');
    
    experimentSelect.addEventListener('change', updateNSelect);
    nSelect.addEventListener('change', updateEpsSelect);
    epsSelect.addEventListener('change', displaySelectedResult);
    
    // Listener para el toggle de persistencia
    showPersistenceToggle.addEventListener('change', togglePersistenceMetrics);
    
    // Intentar cargar estado guardado
    console.log('[INIT] Intentando cargar estado desde localStorage...');
    const wasRestored = loadStateFromLocalStorage();
    if (wasRestored) {
        console.log('[INIT] Estado restaurado desde localStorage');
        showCacheIndicator();
    } else {
        console.log('[INIT] No hay estado guardado, cargando desde GitHub...');
        loadResultsFromGitHub();
    }
    
    // Agregar listener para navegación con teclado
    document.addEventListener('keydown', handleKeyboardNavigation);
});

// Cargar resultados desde GitHub
async function loadResultsFromGitHub() {
    console.log('[GITHUB] Iniciando carga desde GitHub...');
    const content = document.getElementById('content');
    content.innerHTML = '<div class="loading">Cargando resultados desde GitHub...</div>';
    
    // Crear AbortController para poder cancelar
    appState.githubAbortController = new AbortController();
    const signal = appState.githubAbortController.signal;
    console.log('[GITHUB] AbortController creado');
    
    try {
        // URL de la API de GitHub para listar contenidos de la carpeta
        const apiUrl = 'https://api.github.com/repos/habla-liaa/persistent-cost/contents/docs/results';
        console.log('[GITHUB] Fetching:', apiUrl);
        
        const response = await fetch(apiUrl, { signal });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const files = await response.json();
        
        // Filtrar solo archivos JSON
        const jsonFiles = files.filter(file => 
            file.type === 'file' && file.name.endsWith('.json')
        );
        
        console.log('[GITHUB] Archivos JSON encontrados:', jsonFiles.length, jsonFiles.map(f => f.name));
        
        if (jsonFiles.length === 0) {
            console.log('[GITHUB] ERROR: No se encontraron archivos JSON');
            showError('No se encontraron archivos JSON en el repositorio');
            return;
        }
        
        // Cargar cada archivo JSON (pasando el nombre para extraer eps)
        console.log('[GITHUB] Cargando', jsonFiles.length, 'archivos JSON...');
        const promises = jsonFiles.map(file => loadJSONFromURL(file.download_url, file.name, signal));
        const results = await Promise.all(promises);
        console.log('[GITHUB] Archivos cargados:', results.filter(r => r !== null).length, 'exitosos');
        
        // Verificar si fue cancelado antes de actualizar el estado
        if (signal.aborted) {
            console.log('[GITHUB] Carga abortada, no actualizando estado');
            return;
        }
        
        // Filtrar resultados nulos y sin experiment_name (como summary.json)
        appState.loadedResults = results.filter(r => r !== null && r.experiment_name);
        appState.dataSource = 'github';
        console.log('[GITHUB] Resultados válidos:', appState.loadedResults.length);
        
        if (appState.loadedResults.length > 0) {
            console.log('[GITHUB] Actualizando UI...');
            updateDataSourceIndicator();
            populateSelectors();
            document.getElementById('resultsSelector').style.display = 'block';
            saveStateToLocalStorage();
            
            // Mostrar el primer resultado automáticamente
            const firstExperiment = appState.loadedResults[0].experiment_name;
            document.getElementById('experimentSelect').value = firstExperiment;
            updateNSelect();
        } else {
            showError('No se pudieron cargar los resultados desde GitHub');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Carga de GitHub cancelada');
            return;
        }
        console.error('Error cargando desde GitHub:', error);
        showError(`Error cargando resultados: ${error.message}. Puede cargar archivos locales usando el botón "Cargar Carpeta Local".`);
    } finally {
        appState.githubAbortController = null;
    }
}

async function loadJSONFromURL(url, filename = '', signal = null) {
    try {
        const response = await fetch(url, signal ? { signal } : {});
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        // Get text first, then handle Infinity values before parsing
        const text = await response.text();
        // Replace "Infinity" with null (which represents infinite bars)
        const sanitizedText = text.replace(/:\s*Infinity/g, ': null');
        const data = JSON.parse(sanitizedText);
        // Extraer eps del nombre del archivo si no está en el JSON
        if (data.eps === undefined && filename) {
            const epsMatch = filename.match(/_eps([\d.]+)/);
            data.eps = epsMatch ? parseFloat(epsMatch[1]) : 0;
        }
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            return null;
        }
        console.error('Error cargando JSON desde URL:', error);
        return null;
    }
}

function togglePersistenceMetrics() {
    const showPersistence = document.getElementById('showPersistenceMetrics').checked;
    const persistenceRows = document.querySelectorAll('.persistence-metric');
    
    persistenceRows.forEach(row => {
        if (showPersistence) {
            row.classList.add('visible');
        } else {
            row.classList.remove('visible');
        }
    });
}

function handleKeyboardNavigation(event) {
    // Solo si hay un resultado cargado
    if (!appState.currentResult) return;
    
    // Obtener métodos disponibles (excluir 'cylinder' method)
    const methods = ['cone', 'cone2', 'cone_htr', 'cone_gd'].filter(m => 
        appState.currentResult[m] && !appState.currentResult[m].error
    );
    
    if (methods.length === 0) return;
    
    const currentIndex = methods.indexOf(appState.currentMethod);
    
    if (event.key === 'ArrowLeft') {
        event.preventDefault();
        // Ir al método anterior (circular)
        const newIndex = currentIndex > 0 ? currentIndex - 1 : methods.length - 1;
        switchToMethod(methods[newIndex]);
    } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        // Ir al método siguiente (circular)
        const newIndex = currentIndex < methods.length - 1 ? currentIndex + 1 : 0;
        switchToMethod(methods[newIndex]);
    }
}

function switchToMethod(method) {
    appState.currentMethod = method;
    
    // Actualizar tabs visuales
    const tabs = document.querySelectorAll('.method-tab');
    tabs.forEach(tab => {
        if (tab.dataset.method === method) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Mostrar contenido correspondiente
    document.querySelectorAll('.method-content').forEach(c => c.classList.remove('active'));
    const targetContent = document.querySelector(`.method-content[data-method="${method}"]`);
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // Redibujar diagramas
    drawAllDiagrams(appState.currentResult, method);
    
    // Actualizar barra de estado
    updateStatusBar();
    
    // Guardar estado
    saveStateToLocalStorage();
}

function updateStatusBar() {
    const statusBar = document.getElementById('statusBar');
    const statusMethod = document.getElementById('statusMethod');
    const statusExperiment = document.getElementById('statusExperiment');
    const statusN = document.getElementById('statusN');
    
    if (appState.currentResult) {
        statusBar.style.display = 'flex';
        statusMethod.textContent = appState.currentMethod.toUpperCase();
        statusExperiment.textContent = appState.currentResult.experiment_name;
        statusN.textContent = appState.currentResult.n;
    } else {
        statusBar.style.display = 'none';
    }
}

// Manejo de carga de carpeta local
function handleFolderSelect(event) {
    console.log('[LOCAL] Usuario seleccionó carpeta local');
    
    // Cancelar carga de GitHub si está en progreso
    if (appState.githubAbortController) {
        console.log('[LOCAL] Cancelando carga de GitHub en progreso...');
        appState.githubAbortController.abort();
        appState.githubAbortController = null;
        console.log('[LOCAL] Carga de GitHub cancelada');
    }
    
    const allFiles = Array.from(event.target.files);
    console.log('[LOCAL] Total archivos en carpeta:', allFiles.length);
    const files = allFiles.filter(f => f.name.endsWith('.json'));
    
    console.log('[LOCAL] Archivos JSON encontrados:', files.length, files.map(f => f.name));
    
    if (files.length === 0) {
        showError('No se encontraron archivos JSON en la carpeta');
        return;
    }
    loadFiles(files);
}

function loadFiles(files) {
    console.log('[LOCAL] loadFiles: Iniciando carga de', files.length, 'archivos');
    appState.loadedResults = [];
    const promises = [];
    
    for (let file of files) {
        promises.push(loadJSONFile(file));
    }
    
    console.log('[LOCAL] Esperando que todos los archivos se carguen...');
    Promise.all(promises).then(results => {
        console.log('[LOCAL] Todos los archivos procesados');
        // Filtrar resultados nulos y sin experiment_name (como summary.json)
        appState.loadedResults = results.filter(r => r !== null && r.experiment_name);
        appState.dataSource = 'local';
        console.log('[LOCAL] Resultados válidos:', appState.loadedResults.length);
        console.log('[LOCAL] Resultados con eps:', appState.loadedResults.map(r => ({
            name: r.experiment_name,
            n: r.n,
            eps: r.eps
        })));
        if (appState.loadedResults.length > 0) {
            updateDataSourceIndicator();
            populateSelectors();
            document.getElementById('resultsSelector').style.display = 'block';
            saveStateToLocalStorage();
        } else {
            showError('No se pudieron cargar los archivos JSON');
        }
    });
}

function loadJSONFile(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                // Extraer eps del nombre del archivo si no está en el JSON
                if (data.eps === undefined) {
                    const epsMatch = file.name.match(/_eps([\d.]+)/);
                    data.eps = epsMatch ? parseFloat(epsMatch[1]) : 0;
                }
                resolve(data);
            } catch (error) {
                console.error('Error parseando JSON:', error);
                resolve(null);
            }
        };
        reader.onerror = () => resolve(null);
        reader.readAsText(file);
    });
}

// Poblar selectores
function populateSelectors() {
    console.log('[UI] populateSelectors: Poblando selectores...');
    const experimentSelect = document.getElementById('experimentSelect');
    const experiments = [...new Set(appState.loadedResults.map(r => r.experiment_name))].sort();
    console.log('[UI] Experimentos encontrados:', experiments);
    
    experimentSelect.innerHTML = '<option value="">Seleccionar experimento...</option>';
    experiments.forEach(exp => {
        const option = document.createElement('option');
        option.value = exp;
        option.textContent = exp;
        experimentSelect.appendChild(option);
    });
}

function updateNSelect() {
    console.log('[UI] updateNSelect llamado');
    const experimentSelect = document.getElementById('experimentSelect');
    const nSelect = document.getElementById('nSelect');
    const epsSelect = document.getElementById('epsSelect');
    const selectedExp = experimentSelect.value;
    console.log('[UI] Experimento seleccionado:', selectedExp);
    
    if (!selectedExp) {
        nSelect.innerHTML = '<option value="">Seleccionar n...</option>';
        nSelect.disabled = true;
        epsSelect.style.display = 'none';
        return;
    }
    
    // Obtener valores de n únicos para este experimento
    const nValues = [...new Set(appState.loadedResults
        .filter(r => r.experiment_name === selectedExp)
        .map(r => r.n))]
        .sort((a, b) => a - b);
    
    nSelect.innerHTML = '<option value="">Seleccionar n...</option>';
    nValues.forEach(n => {
        const option = document.createElement('option');
        option.value = n;
        option.textContent = `n = ${n}`;
        nSelect.appendChild(option);
    });
    nSelect.disabled = false;
    
    // Seleccionar automáticamente el n más pequeño
    if (nValues.length > 0) {
        nSelect.value = nValues[0];
        updateEpsSelect();
    }
    
    // Guardar estado
    saveStateToLocalStorage();
}

function updateEpsSelect() {
    console.log('[UI] updateEpsSelect llamado');
    const experimentSelect = document.getElementById('experimentSelect');
    const nSelect = document.getElementById('nSelect');
    const epsSelect = document.getElementById('epsSelect');
    const selectedExp = experimentSelect.value;
    const selectedN = parseInt(nSelect.value);
    console.log('[UI] Experimento:', selectedExp, 'N:', selectedN);
    
    if (!selectedExp || !selectedN) {
        epsSelect.innerHTML = '<option value="">Seleccionar ε...</option>';
        epsSelect.disabled = true;
        return;
    }
    
    // Filtrar resultados por experimento y n
    const matchingResults = appState.loadedResults.filter(
        r => r.experiment_name === selectedExp && r.n === selectedN
    );
    console.log('[UI] Resultados que coinciden:', matchingResults.length);
    
    // Extraer valores de eps (siempre existe, default es 0)
    const epsValues = [...new Set(matchingResults.map(r => r.eps ?? 0))].sort((a, b) => a - b);
    console.log('[UI] Valores de eps encontrados:', epsValues);
    
    // Actualizar selector de eps
    epsSelect.disabled = false;
    epsSelect.innerHTML = '';
    
    if (epsValues.length === 0) {
        epsSelect.innerHTML = '<option value="">Sin resultados</option>';
        epsSelect.disabled = true;
    } else {
        epsValues.forEach(eps => {
            const option = document.createElement('option');
            option.value = eps;
            option.textContent = `ε = ${eps}`;
            epsSelect.appendChild(option);
        });
        
        // Seleccionar el primer valor automáticamente
        epsSelect.value = epsValues[0];
        displaySelectedResult();
    }
    
    // Guardar estado
    saveStateToLocalStorage();
}

function displaySelectedResult() {
    console.log('[UI] displaySelectedResult llamado');
    const experimentSelect = document.getElementById('experimentSelect');
    const nSelect = document.getElementById('nSelect');
    const epsSelect = document.getElementById('epsSelect');
    const selectedExp = experimentSelect.value;
    const selectedN = parseInt(nSelect.value);
    const selectedEps = parseFloat(epsSelect.value);
    console.log('[UI] Buscando resultado - Exp:', selectedExp, 'N:', selectedN, 'Eps:', selectedEps);
    
    if (!selectedExp || !selectedN || isNaN(selectedEps)) {
        console.log('[UI] displaySelectedResult: Parámetros incompletos, saliendo');
        return;
    }
    
    // Buscar resultado que coincida con experimento, n y eps
    const result = appState.loadedResults.find(r => {
        const matchExp = r.experiment_name === selectedExp;
        const matchN = r.n === selectedN;
        const matchEps = (r.eps ?? 0) === selectedEps;
        return matchExp && matchN && matchEps;
    });
    
    if (result) {
        console.log('[UI] Resultado encontrado:', result.experiment_name, 'n='+result.n, 'eps='+result.eps);
        appState.currentResult = result;
        renderResult(result);
        saveStateToLocalStorage();
    } else {
        console.log('[UI] ERROR: No se encontró resultado para los parámetros dados');
        console.log('[UI] Buscando en', appState.loadedResults.length, 'resultados disponibles');
    }
}

// Renderizado principal
function renderResult(result) {
    const content = document.getElementById('content');
    
    // Panel de información
    const infoHTML = `
        <div class="info-panel">
            <div class="info-grid">
                <div class="info-card">
                    <h3>Experimento</h3>
                    <p>${result.experiment_name}</p>
                </div>
                <div class="info-card">
                    <h3>Tamaño (n)</h3>
                    <p>${result.n}</p>
                </div>
                <div class="info-card">
                    <h3>Dimensión</h3>
                    <p>${result.dim || 'N/A'}</p>
                </div>
                <div class="info-card">
                    <h3>Lipschitz</h3>
                    <p>${result.lipschitz_constant.toFixed(4)}</p>
                </div>
                ${result.eps !== undefined ? `
                <div class="info-card">
                    <h3>Epsilon (ε)</h3>
                    <p>${result.eps}</p>
                </div>
                ` : ''}
                ${result.seed !== undefined ? `
                <div class="info-card">
                    <h3>Seed</h3>
                    <p>${result.seed}</p>
                </div>
                ` : ''}
            </div>
            
            <div class="method-tabs">
                ${generateMethodTabs(result)}
            </div>
            
            <div id="methodContents">
                ${generateMethodContents(result)}
            </div>
        </div>
    `;
    
    content.innerHTML = infoHTML;
    
    // Activar tabs
    setupMethodTabs();
    
    // Dibujar diagramas del método activo
    drawAllDiagrams(result, appState.currentMethod);
    
    // Actualizar barra de estado
    updateStatusBar();
    
    // Guardar estado
    saveStateToLocalStorage();
}

function generateMethodTabs(result) {
    // Solo mostrar cone, cone2, cone_htr y cone_gd, no cylinder method
    const methods = ['cone', 'cone2', 'cone_htr', 'cone_gd'].filter(m => result[m] && !result[m].error);
    return methods.map(method => `
        <button class="method-tab ${method === appState.currentMethod ? 'active' : ''}" 
                data-method="${method}">
            ${method.toUpperCase()}
        </button>`).join('');
}

function generateMethodContents(result) {
    // Solo mostrar cone, cone2, cone_htr y cone_gd, no cylinder method
    const methods = ['cone', 'cone2', 'cone_htr', 'cone_gd'].filter(m => result[m] && !result[m].error);
    return methods.map(method => {
        const methodData = result[method];
        return `
            <div class="method-content ${method === appState.currentMethod ? 'active' : ''}" 
                 data-method="${method}">
                ${generateDiagramsGrid(methodData, method)}
            </div>
        `;
    }).join('');
}

function generateDiagramsGrid(methodData, method) {
    let html = '';
    
    // Botón para abrir modal de nubes de puntos
    if (methodData.X && methodData.Y) {
        const dimX = methodData.X[0] ? methodData.X[0].length : 0;
        const dimY = methodData.Y[0] ? methodData.Y[0].length : 0;
        if ((dimX === 2 || dimX === 3) && (dimY === 2 || dimY === 3)) {
            html += '<div style="margin: 20px 0; text-align: center;">';
            html += `<a class="pointcloud-link" onclick="openPointCloudModal('${method}')">`;
            html += `📊 Ver Nubes de Puntos (X: ${dimX}D, Y: ${dimY}D)`;
            html += '</a>';
            html += '</div>';
        }
    }
    
    // Primera fila: X, Y, Cylinder
    html += '<h3 style="margin: 20px 0 10px 0; color: #2c3e50; font-size: 1.1em;">Diagramas de Persistencia</h3>';
    html += '<div class="diagrams-grid persistence-row">';
    html += generateDiagramCard({ key: 'dgm_X', title: 'Espacio X', id: `${method}-X` }, methodData);
    html += generateDiagramCard({ key: 'dgm_Y', title: 'Espacio Y', id: `${method}-Y` }, methodData);
    html += generateDiagramCard({ key: 'dgm_cylinder', title: 'Cilindro', id: `${method}-cylinder` }, methodData);
    html += '</div>';
    
    // Segunda fila: Cone, Ker, Coker
    html += '<div class="diagrams-grid persistence-row">';
    html += generateDiagramCard({ key: 'dgm_cone', title: 'Cono', id: `${method}-cone` }, methodData);
    html += generateDiagramCard({ key: 'dgm_ker', title: 'Kernel', id: `${method}-ker` }, methodData);
    html += generateDiagramCard({ key: 'dgm_coker', title: 'Cokernel', id: `${method}-coker` }, methodData);
    html += '</div>';
    
    // Tercera fila: Missing (si existe)
    if (methodData.missing && methodData.missing.length > 0) {
        html += '<div class="diagrams-grid missing-row">';
        html += generateDiagramCard({ key: 'missing', title: 'Missing', id: `${method}-missing` }, methodData);
        html += '</div>';
    }
    
    return html;
}

function generatePointCloudCard(points, title, id) {
    if (!points || points.length === 0) return '';
    
    const dim = points[0].length;
    if (dim < 2 || dim > 3) return '';
    
    return `
        <div class="diagram-card">
            <h3>${title} <span class="badge" style="background: #34495e; color: white;">${dim}D</span></h3>
            <div class="pointcloud-canvas" id="${id}"></div>
            <div class="stats-table">
                <table>
                    <tbody>
                        <tr><td>Puntos</td><td>${points.length}</td></tr>
                        <tr><td>Dimensión</td><td>${dim}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

function generateDiagramCard(diagram, methodData) {
    const dgm = methodData[diagram.key];
    const stats = computeStatistics(dgm);
    
    return `
        <div class="diagram-card">
            <h3>
                ${diagram.title}
                ${generateDimensionBadges(dgm)}
            </h3>
            <canvas class="diagram-canvas" id="${diagram.id}"></canvas>
            ${generateStatsTable(stats)}
            ${generateBarsList(dgm, diagram.title)}
        </div>
    `;
}

function generateDimensionBadges(dgm) {
    if (!dgm) return '';
    const counts = countBarsByDimension(dgm);
    return counts.map((count, dim) => 
        count > 0 ? `<span class="badge badge-h${dim}">${DIM_LABELS[dim]}: ${count}</span>` : ''
    ).join('');
}

function generateStatsTable(stats) {
    return `
        <div class="stats-table">
            <table>
                <thead>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Total barras</td><td>${stats.n_bars}</td></tr>
                    <tr><td>Barras finitas</td><td>${stats.n_finite}</td></tr>
                    <tr><td>Barras infinitas</td><td>${stats.n_infinite}</td></tr>
                    <tr class="persistence-metric"><td>Persistencia total</td><td>${stats.total_persistence.toFixed(4)}</td></tr>
                    <tr class="persistence-metric"><td>Persistencia promedio</td><td>${stats.avg_persistence.toFixed(4)}</td></tr>
                    <tr class="persistence-metric"><td>Persistencia máxima</td><td>${stats.max_persistence.toFixed(4)}</td></tr>
                </tbody>
            </table>
        </div>
    `;
}

function generateBarsList(dgm, title) {
    if (!dgm || dgm.length === 0) return '';
    
    let html = '<div class="bars-list"><h4>Lista de Barras</h4><div class="bars-container">';
    
    dgm.forEach((dimBars, dim) => {
        if (!dimBars || dimBars.length === 0) return;
        
        html += `<div class="dimension-section">`;
        html += `<div class="dimension-header">${DIM_LABELS[dim]} (${dimBars.length} barras)</div>`;
        
        dimBars.forEach((bar, idx) => {
            // Verificar si estamos en vista "Missing" con formato (categoria, (birth, death))
            if (title === 'Missing' && Array.isArray(bar) && bar.length === 2 && typeof bar[0] === 'string') {
                const category = bar[0];
                const coords = bar[1];
                
                if (!Array.isArray(coords) || coords.length < 2) return;
                
                const b = typeof coords[0] === 'number' ? coords[0] : parseFloat(coords[0]);
                const d = coords[1];
                
                if (isNaN(b)) return;
                
                const birth = b.toFixed(4);
                const isInfinite = (d === null || !isFinite(d));
                const death = isInfinite ? '∞' : (typeof d === 'number' ? d.toFixed(4) : parseFloat(d).toFixed(4));
                const pers = isInfinite ? '∞' : (d - b).toFixed(4);
                html += `<div class="bar-item">[${idx}] <span style="color: #e67e22; font-weight: bold;">${category}</span>: (${birth}, ${death}) — pers: ${pers}</div>`;
            } else {
                // Formato normal [birth, death]
                if (!Array.isArray(bar) || bar.length < 2) return;
                
                const b = typeof bar[0] === 'number' ? bar[0] : parseFloat(bar[0]);
                const d = bar[1];
                
                if (isNaN(b)) return;
                
                const birth = b.toFixed(4);
                // Manejar null como infinito
                const isInfinite = (d === null || !isFinite(d));
                const death = isInfinite ? '∞' : (typeof d === 'number' ? d.toFixed(4) : parseFloat(d).toFixed(4));
                const pers = isInfinite ? '∞' : (d - b).toFixed(4);
                html += `<div class="bar-item">[${idx}] (${birth}, ${death}) — pers: ${pers}</div>`;
            }
        });
        
        html += `</div>`;
    });
    
    html += '</div></div>';
    return html;
}

// Configurar tabs
function setupMethodTabs() {
    const tabs = document.querySelectorAll('.method-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const method = tab.dataset.method;
            switchToMethod(method);
        });
    });
}

// Dibujar diagramas
function drawAllDiagrams(result, method) {
    const methodData = result[method];
    if (!methodData || methodData.error) return;
    
    // Calcular rango global para todos los diagramas
    const globalRange = calculateGlobalRange(methodData);
    
    // Esperar a que el DOM esté actualizado
    setTimeout(() => {
        // Dibujar diagramas de persistencia con rango global
        drawDiagram(`${method}-X`, methodData.dgm_X, globalRange);
        drawDiagram(`${method}-Y`, methodData.dgm_Y, globalRange);
        drawDiagram(`${method}-cylinder`, methodData.dgm_cylinder, globalRange);
        drawDiagram(`${method}-cone`, methodData.dgm_cone, globalRange);
        drawDiagram(`${method}-ker`, methodData.dgm_ker, globalRange);
        drawDiagram(`${method}-coker`, methodData.dgm_coker, globalRange);
        if (methodData.missing && methodData.missing.length > 0) {
            drawDiagram(`${method}-missing`, methodData.missing, globalRange);
        }
    }, 10);
}

function calculateGlobalRange(methodData) {
    const diagrams = [
        methodData.dgm_X,
        methodData.dgm_Y,
        methodData.dgm_cylinder,
        methodData.dgm_cone,
        methodData.dgm_ker,
        methodData.dgm_coker,
        methodData.missing
    ].filter(d => d);
    
    let max = 0;
    diagrams.forEach(dgm => {
        if (!dgm) return;
        const range = calculateRange(dgm);
        max = Math.max(max, range.max);
    });
    
    return { max: max === 0 ? 1 : max };
}

function drawPointCloud(divId, points, title = '') {
    const div = document.getElementById(divId);
    if (!div || !points || points.length === 0) return;
    
    const dim = points[0].length;
    
    if (dim === 2) {
        // 2D scatter plot
        const trace = {
            x: points.map(p => p[0]),
            y: points.map(p => p[1]),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 8,
                color: '#3498db',
                opacity: 0.7
            },
            name: title
        };
        
        const layout = {
            title: title ? { text: title, font: { size: 16 } } : undefined,
            margin: { l: 50, r: 30, t: title ? 50 : 30, b: 50 },
            xaxis: { title: 'x₁', showgrid: true, zeroline: true },
            yaxis: { title: 'x₂', showgrid: true, zeroline: true },
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: '#fafafa',
            showlegend: false
        };
        
        Plotly.newPlot(div, [trace], layout, { responsive: true, displayModeBar: true });
        
    } else if (dim === 3) {
        // 3D scatter plot
        const trace = {
            x: points.map(p => p[0]),
            y: points.map(p => p[1]),
            z: points.map(p => p[2]),
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 5,
                color: '#3498db',
                opacity: 0.7
            },
            name: title
        };
        
        const layout = {
            title: title ? { text: title, font: { size: 16 } } : undefined,
            margin: { l: 0, r: 0, t: title ? 50 : 0, b: 0 },
            scene: {
                xaxis: { title: 'x₁', showgrid: true },
                yaxis: { title: 'x₂', showgrid: true },
                zaxis: { title: 'x₃', showgrid: true },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: '#fafafa',
            showlegend: false
        };
        
        Plotly.newPlot(div, [trace], layout, { responsive: true, displayModeBar: true });
    }
}

function drawDiagram(canvasId, dgm, globalRange) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Limpiar
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    if (!dgm || dgm.length === 0) {
        ctx.fillStyle = '#6c757d';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Sin datos', width/2, height/2);
        return;
    }
    
    // Usar rango global si se proporciona, sino calcular local
    const range = globalRange || calculateRange(dgm);
    if (range.max === 0) range.max = 1;
    
    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;
    
    // Función para escalar coordenadas
    const scaleX = x => padding + (x / (range.max * 1.1)) * plotWidth;
    const scaleY = y => height - padding - (y / (range.max * 1.1)) * plotHeight;
    
    // Dibujar ejes
    ctx.strokeStyle = '#dee2e6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(padding, padding);
    ctx.stroke();
    
    // Dibujar diagonal
    ctx.strokeStyle = '#adb5bd';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(scaleX(0), scaleY(0));
    ctx.lineTo(scaleX(range.max * 1.1), scaleY(range.max * 1.1));
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Dibujar puntos por dimensión
    dgm.forEach((dimBars, dim) => {
        if (!dimBars || dimBars.length === 0) return;
        
        ctx.fillStyle = DIMENSION_COLORS[dim % DIMENSION_COLORS.length];
        
        dimBars.forEach(bar => {
            // Validar que bar sea un array con elementos
            if (!Array.isArray(bar) || bar.length < 2) return;
            
            let birth, death, isMissingFormat = false;
            
            // Detectar formato missing: (categoria, (birth, death))
            if (typeof bar[0] === 'string' && Array.isArray(bar[1]) && bar[1].length >= 2) {
                isMissingFormat = true;
                const coords = bar[1];
                birth = typeof coords[0] === 'number' ? coords[0] : parseFloat(coords[0]);
                death = coords[1];
            } else {
                // Formato normal: [birth, death]
                birth = typeof bar[0] === 'number' ? bar[0] : parseFloat(bar[0]);
                death = bar[1];
            }
            
            if (isNaN(birth)) return;
            
            // Manejar null como infinito
            const isInfinite = (death === null || !isFinite(death));
            
            if (!isInfinite) {
                // Barra finita
                const x = scaleX(birth);
                const y = scaleY(death);
                
                if (isMissingFormat) {
                    // Para missing, dibujar cuadrados en lugar de círculos
                    ctx.fillRect(x - 4, y - 4, 8, 8);
                } else {
                    // Barra normal - círculo
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fill();
                }
            } else {
                // Barra infinita - triángulo
                const x = scaleX(birth);
                const y = padding + 10;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x - 5, y + 8);
                ctx.lineTo(x + 5, y + 8);
                ctx.closePath();
                ctx.fill();
            }
        });
    });
    
    // Etiquetas de ejes
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Birth', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Death', 0, 0);
    ctx.restore();
    
    // Valores en ejes
    ctx.font = '10px Arial';
    ctx.fillStyle = '#6c757d';
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
        const val = (range.max * 1.1 * i / ticks);
        const label = val.toFixed(2);
        
        // Eje X
        const x = scaleX(val);
        ctx.textAlign = 'center';
        ctx.fillText(label, x, height - padding + 20);
        
        // Eje Y
        const y = scaleY(val);
        ctx.textAlign = 'right';
        ctx.fillText(label, padding - 10, y + 4);
    }
}

// Funciones auxiliares
function calculateRange(dgm) {
    let max = 0;
    dgm.forEach(dimBars => {
        if (!dimBars) return;
        dimBars.forEach(bar => {
            // Validar que bar sea un array válido
            if (!Array.isArray(bar) || bar.length < 2) return;
            
            let b, d;
            
            // Detectar formato missing: (categoria, (birth, death))
            if (typeof bar[0] === 'string' && Array.isArray(bar[1]) && bar[1].length >= 2) {
                const coords = bar[1];
                b = typeof coords[0] === 'number' ? coords[0] : parseFloat(coords[0]);
                d = coords[1];
            } else {
                // Formato normal: [birth, death]
                b = typeof bar[0] === 'number' ? bar[0] : parseFloat(bar[0]);
                d = bar[1];
            }
            
            if (!isNaN(b)) {
                max = Math.max(max, b);
            }
            // Ignorar null/infinito al calcular rango
            if (d !== null && typeof d === 'number' && isFinite(d)) {
                max = Math.max(max, d);
            } else if (d !== null && typeof d !== 'number') {
                const dNum = parseFloat(d);
                if (!isNaN(dNum) && isFinite(dNum)) {
                    max = Math.max(max, dNum);
                }
            }
        });
    });
    return { max };
}

function computeStatistics(dgm) {
    const stats = {
        n_bars: 0,
        n_finite: 0,
        n_infinite: 0,
        total_persistence: 0,
        avg_persistence: 0,
        max_persistence: 0
    };
    
    if (!dgm) return stats;
    
    const allBars = [];
    dgm.forEach(dimBars => {
        if (dimBars) allBars.push(...dimBars);
    });
    
    stats.n_bars = allBars.length;
    
    allBars.forEach(bar => {
        // Validar que bar sea un array válido
        if (!Array.isArray(bar) || bar.length < 2) return;
        
        let b, d;
        
        // Detectar formato missing: (categoria, (birth, death))
        if (typeof bar[0] === 'string' && Array.isArray(bar[1]) && bar[1].length >= 2) {
            const coords = bar[1];
            b = typeof coords[0] === 'number' ? coords[0] : parseFloat(coords[0]);
            d = coords[1];
        } else {
            // Formato normal: [birth, death]
            b = typeof bar[0] === 'number' ? bar[0] : parseFloat(bar[0]);
            d = bar[1];
        }
        
        if (isNaN(b)) return;
        
        // Manejar null como infinito
        const isInfinite = (d === null || !isFinite(d));
        
        if (!isInfinite) {
            stats.n_finite++;
            const dNum = typeof d === 'number' ? d : parseFloat(d);
            if (!isNaN(dNum)) {
                const pers = dNum - b;
                stats.total_persistence += pers;
                stats.max_persistence = Math.max(stats.max_persistence, pers);
            }
        } else {
            stats.n_infinite++;
        }
    });
    
    if (stats.n_finite > 0) {
        stats.avg_persistence = stats.total_persistence / stats.n_finite;
    }
    
    return stats;
}

function countBarsByDimension(dgm) {
    if (!dgm) return [];
    return dgm.map(dimBars => dimBars ? dimBars.length : 0);
}

function showError(message) {
    const content = document.getElementById('content');
    content.innerHTML = `<div class="error">${message}</div>`;
}

// Modal functions
function openPointCloudModal(method) {
    const methodData = appState.currentResult[method];
    if (!methodData || !methodData.X || !methodData.Y) return;
    
    const modal = document.getElementById('pointCloudModal');
    const modalTitle = document.getElementById('modalTitle');
    
    modalTitle.textContent = `Nubes de Puntos - ${method.toUpperCase()}`;
    modal.classList.add('active');
    
    // Esperar a que el modal esté visible para dibujar
    setTimeout(() => {
        drawPointCloud('modalPlotX', methodData.X, 'Espacio X');
        drawPointCloud('modalPlotY', methodData.Y, 'Espacio Y');
    }, 50);
    
    // Cerrar con ESC
    document.addEventListener('keydown', handleModalEscape);
}

function closePointCloudModal() {
    const modal = document.getElementById('pointCloudModal');
    modal.classList.remove('active');
    document.removeEventListener('keydown', handleModalEscape);
}

function handleModalEscape(event) {
    if (event.key === 'Escape') {
        closePointCloudModal();
    }
}

// Cerrar modal al hacer clic fuera del contenido
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('pointCloudModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closePointCloudModal();
            }
        });
    }
});
