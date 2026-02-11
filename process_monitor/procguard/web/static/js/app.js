const socket = io();
let workersData = {};
let healthData = {};
let workerGroups = {};
let selectedWorkers = new Set();
let groupConfigModalOpen = false;
let domContentLoaded = false;

let createGroupModal, moveGroupModal, logModal, groupConfigModal;

let refreshWorkersTimer = null;
let cleanupMemoryTimer = null;
const MAX_LOGS = 200;
const MAX_WORKERS_DATA = 100;
const MAX_HEALTH_DATA = 100;

let serverStartupTime = null;
let serverVersion = null;
let lastDisconnectTime = null;

document.addEventListener('DOMContentLoaded', () => {
    domContentLoaded = true;
    loadInitialData();
    const loaded = loadGroupConfig();
    if (loaded) {
        renderWorkerGroups();
    }
});

function initModals() {
    const createGroupModalEl = document.getElementById('createGroupModal');
    const moveGroupModalEl = document.getElementById('moveGroupModal');
    const logModalEl = document.getElementById('logModal');
    const groupConfigModalEl = document.getElementById('groupConfigModal');
    
    if (createGroupModalEl) {
        createGroupModal = new bootstrap.Modal(createGroupModalEl);
    }
    if (moveGroupModalEl) {
        moveGroupModal = new bootstrap.Modal(moveGroupModalEl);
    }
    if (logModalEl) {
        logModal = new bootstrap.Modal(logModalEl);
    }
    if (groupConfigModalEl) {
        groupConfigModal = new bootstrap.Modal(groupConfigModalEl, {
            backdrop: 'static',
            keyboard: false
        });
        
        groupConfigModalEl.addEventListener('hidden.bs.modal', () => {
            groupConfigModalOpen = false;
        });
    }
}

socket.on('connect', () => {
    updateConnectionStatus(true);
    socket.emit('subscribe', {});
    initModals();
    
    if (!domContentLoaded) {
        const checkDom = setInterval(() => {
            if (domContentLoaded) {
                clearInterval(checkDom);
                setTimeout(initWorkerGroupsIfNeeded, 50);
            }
        }, 10);
    } else {
        initWorkerGroupsIfNeeded();
    }

    initWorkerEnvModal();
    
    checkServerRestart();
    
    if (lastDisconnectTime && Date.now() - lastDisconnectTime > 5000) {
        console.log('[Server] 检测到长时间断开后的重连，可能是服务器重启');
        handleServerRestart();
    }
});

socket.on('disconnect', () => {
    console.log('Disconnected from ProcGuard server');
    updateConnectionStatus(false);
    lastDisconnectTime = Date.now();
    
    setTimeout(() => {
        console.log('[WebSocket] 尝试重新连接...');
        socket.connect();
    }, 3000);
});

socket.on('status_update', (data) => {
    console.log('Status update:', data);
    updateDashboard(data);
});

socket.on('worker_update', (data) => {
    if (data.action === 'registered') {
        workersData[data.worker_id] = {
            status: 'stopped',
            pid: null,
            restart_count: 0,
            command: data.command
        };
        
        if (data.is_reregistration) {
            console.log(`[Worker] Worker ${data.worker_id} 重新注册，清除旧数据`);
            delete workersData[data.worker_id];
            workersData[data.worker_id] = {
                status: 'stopped',
                pid: null,
                restart_count: 0,
                command: data.command
            };
        }
        
        addWorkerRow(data.worker_id, workersData[data.worker_id]);

        if (!workerGroups || !workerGroups['default']) {
            createDefaultGroup();
        }

        if (workerGroups && workerGroups['default'] && !workerGroups['default'].workers.includes(data.worker_id)) {
            workerGroups['default'].workers.push(data.worker_id);
            renderWorkerGroups();
            saveWorkerGroupsConfig();
        }
        return;
    }

    if (data.action === 'unregistered') {
        console.log(`[Worker] Worker ${data.worker_id} 已注销，清除数据`);
        delete workersData[data.worker_id];
        cleanupRemovedWorkerFromGroups(data.worker_id);
        removeWorkerRow(data.worker_id);
        updateGroupStats();
        return;
    }

    if (data.action === 'removed' || data.action === 'status_changed') {
        if (data.status === 'stopped' || data.action === 'removed') {
            cleanupRemovedWorkerFromGroups(data.worker_id);
        }
    }

    updateWorkerRow(data.worker_id, data);
    updateWorkerItem(data.worker_id, data);
    updateGroupStats();
});

function cleanupRemovedWorkerFromGroups(workerId) {
    let cleaned = false;

    for (const groupId of Object.keys(workerGroups)) {
        const group = workerGroups[groupId];
        if (group && group.workers) {
            const idx = group.workers.indexOf(workerId);
            if (idx > -1) {
                group.workers.splice(idx, 1);
                cleaned = true;
                console.log(`[WorkerGroups] 从分组 ${groupId} 移除已停止的 worker: ${workerId}`);
            }
        }
    }

    if (cleaned) {
        renderWorkerGroups();
        saveWorkerGroupsConfig();
    }
}

socket.on('recovery_event', (data) => {
    console.log('Recovery event:', data);
});

socket.on('group_cache_cleared', (data) => {
    console.log('Group cache cleared:', data);
    localStorage.removeItem('procguard_groups');
    workerGroups = {};
    renderWorkerGroups();
    alert('分组缓存已清除');
});

async function checkServerRestart() {
    try {
        const response = await fetch('/api/server/info');
        const data = await response.json();
        
        if (serverStartupTime && serverStartupTime !== data.startup_time) {
            console.log('[Server] 检测到服务器重启');
            serverStartupTime = data.startup_time;
            handleServerRestart();
        } else if (!serverStartupTime) {
            serverStartupTime = data.startup_time;
        }
        
        if (serverVersion && serverVersion !== data.version) {
            console.log('[Server] 检测到服务器版本变化');
            serverVersion = data.version;
        } else if (!serverVersion) {
            serverVersion = data.version;
        }
    } catch (error) {
        console.error('[Server] 检查服务器状态失败:', error);
    }
}

function handleServerRestart() {
    console.log('[Server] 处理服务器重启，清除所有缓存');
    localStorage.removeItem('procguard_groups');
    workerGroups = {};
    workersData = {};
    createDefaultGroup();
    refreshWorkers();
}

async function resetAllWorkers() {
    try {
        const response = await fetch('/api/workers/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`[Workers] 重置所有workers: ${result.removed_count} 个worker被移除`);
            
            localStorage.removeItem('procguard_groups');
            workerGroups = {};
            workersData = {};
            
            createDefaultGroup();
            renderWorkerGroups();
            
            alert(`所有workers已重置 (${result.removed_count} 个worker被移除)`);
        } else {
            console.error('[Workers] 重置workers失败:', result.error);
            alert('重置workers失败: ' + (result.error || '未知错误'));
        }
    } catch (error) {
        console.error('[Workers] 重置workers请求失败:', error);
        alert('重置workers请求失败');
    }
}

async function initWorkerGroupsIfNeeded() {
    if (workerGroups && Object.keys(workerGroups).length > 0) {
        return;
    }
    
    console.log('[WorkerGroups] 跳过本地缓存读取，从服务器同步分组数据');
    createDefaultGroup();
}

function validateWorkerDataConsistency() {
    const serverWorkerIds = new Set(Object.keys(workersData));
    let hasIssues = false;
    
    const workerToGroups = {};
    for (const groupId of Object.keys(workerGroups)) {
        const group = workerGroups[groupId];
        if (!group || !group.workers) continue;
        
        for (const workerId of group.workers) {
            if (!workerToGroups[workerId]) {
                workerToGroups[workerId] = [];
            }
            workerToGroups[workerId].push(groupId);
        }
    }
    
    const duplicateWorkers = Object.keys(workerToGroups).filter(w => workerToGroups[w].length > 1);
    if (duplicateWorkers.length > 0) {
        console.warn(`[Validation] 检测到 ${duplicateWorkers.length} 个跨分组重复的workers:`, duplicateWorkers);
        hasIssues = true;
        
        for (const workerId of duplicateWorkers) {
            const groups = workerToGroups[workerId];
            const keepGroup = groups[0];
            const removeGroups = groups.slice(1);
            console.log(`[Validation] Worker ${workerId} 保留在分组 ${keepGroup}，从 ${removeGroups} 中移除`);
            
            for (const groupId of removeGroups) {
                const group = workerGroups[groupId];
                if (group && group.workers) {
                    const idx = group.workers.indexOf(workerId);
                    if (idx > -1) {
                        group.workers.splice(idx, 1);
                        console.log(`[Validation] 从分组 ${groupId} 移除重复worker ${workerId}`);
                    }
                }
            }
        }
    }
    
    for (const groupId of Object.keys(workerGroups)) {
        const group = workerGroups[groupId];
        if (!group || !group.workers) continue;
        
        const invalidWorkers = [];
        const groupDuplicateWorkers = new Set();
        const workerSet = new Set();
        
        for (const workerId of group.workers) {
            if (!serverWorkerIds.has(workerId)) {
                invalidWorkers.push(workerId);
                hasIssues = true;
            }
            
            if (workerSet.has(workerId)) {
                groupDuplicateWorkers.add(workerId);
                hasIssues = true;
            }
            workerSet.add(workerId);
        }
        
        if (invalidWorkers.length > 0) {
            console.warn(`[Validation] 分组 ${groupId} 发现 ${invalidWorkers.length} 个无效workers (未注册):`, invalidWorkers);
            group.workers = group.workers.filter(w => serverWorkerIds.has(w));
        }
        
        if (groupDuplicateWorkers.size > 0) {
            console.warn(`[Validation] 分组 ${groupId} 发现 ${groupDuplicateWorkers.size} 个重复workers:`, Array.from(groupDuplicateWorkers));
            group.workers = [...new Set(group.workers)];
        }
    }
    
    if (hasIssues) {
        console.log('[Validation] 数据一致性检查完成，发现并修复了问题');
        saveWorkerGroupsConfig();
    }
}

function createDefaultGroup() {
    workerGroups = {
        'default': {
            id: 'default',
            name: '默认分组',
            workers: []
        }
    };
    
    for (const workerId of Object.keys(workersData)) {
        workerGroups['default'].workers.push(workerId);
    }
    
    console.log('[WorkerGroups] 创建默认分组，添加所有 workers:', Object.keys(workersData));
    renderWorkerGroups();
    syncGroupsToServer();
}

async function refreshWorkers() {
    try {
        const workersResponse = await fetch('/api/workers');
        const workers = await workersResponse.json();

        const workerIds = Object.keys(workers);
        console.log('[WorkerGroups] 从 /api/workers 获取到 workers:', workerIds);

        for (const [workerId, worker] of Object.entries(workers)) {
            workersData[workerId] = worker;
        }

        const groupsResponse = await fetch('/api/groups');
        const groupsData = await groupsResponse.json();

        if (groupsData.success && groupsData.groups && groupsData.groups.length > 0) {
            console.log('[WorkerGroups] 从服务器同步分组信息');

            const serverGroups = {};
            for (const group of groupsData.groups) {
                serverGroups[group.group_id] = {
                    id: group.group_id,
                    name: group.name || group.group_id,
                    workers: group.workers || [],
                    pytorch_config: group.config || {}
                };
            }

            for (const groupId of Object.keys(serverGroups)) {
                const serverGroup = serverGroups[groupId];
                const validWorkers = serverGroup.workers.filter(w => workersData[w]);
                if (validWorkers.length !== serverGroup.workers.length) {
                    console.log(`[WorkerGroups] 分组 ${groupId} 清理了 ${serverGroup.workers.length - validWorkers.length} 个无效 workers`);
                }
                serverGroup.workers = validWorkers;
            }

            workerGroups = serverGroups;
            console.log('[WorkerGroups] 同步后分组:', Object.keys(workerGroups));
            
            validateWorkerDataConsistency();
            
            renderWorkerGroups();
            saveWorkerGroupsConfig();
            
            if (groupsData.server_info) {
                if (serverStartupTime && serverStartupTime !== groupsData.server_info.startup_time) {
                    console.log('[Server] 检测到服务器重启（通过分组API）');
                    serverStartupTime = groupsData.server_info.startup_time;
                } else if (!serverStartupTime) {
                    serverStartupTime = groupsData.server_info.startup_time;
                }
            }
            return;
        }

        if (groupsData.success && (!groupsData.groups || groupsData.groups.length === 0)) {
            console.warn('[WorkerGroups] 后端返回空分组，可能是重启，清除本地缓存');
            localStorage.removeItem('procguard_groups');
            workerGroups = {};
            createDefaultGroup();
            return;
        }

        if (!workerGroups || Object.keys(workerGroups).length === 0) {
            console.log('[WorkerGroups] workerGroups 为空，创建默认分组...');
            createDefaultGroup();
            return;
        }

        for (const groupId of Object.keys(workerGroups)) {
            const group = workerGroups[groupId];
            if (group && group.workers) {
                group.workers = group.workers.filter(w => workersData[w]);
            }
        }

        const allGroupedWorkers = new Set();
        const duplicateWorkers = new Set();

        for (const groupId of Object.keys(workerGroups)) {
            if (workerGroups[groupId] && workerGroups[groupId].workers) {
                for (const workerId of workerGroups[groupId].workers) {
                    if (allGroupedWorkers.has(workerId)) {
                        duplicateWorkers.add(workerId);
                        console.warn('[WorkerGroups] 发现重复的 worker:', workerId);
                    }
                    allGroupedWorkers.add(workerId);
                }
            }
        }

        if (duplicateWorkers.size > 0) {
            console.warn('[WorkerGroups] 检测到重复 workers，正在清理:', Array.from(duplicateWorkers));
            const cleanedWorkerGroups = { 'default': workerGroups['default'] };

            for (const groupId of Object.keys(workerGroups)) {
                if (groupId === 'default') continue;
                const group = workerGroups[groupId];
                if (group && group.workers) {
                    const uniqueWorkers = group.workers.filter(w => !duplicateWorkers.has(w));
                    if (uniqueWorkers.length > 0) {
                        cleanedWorkerGroups[groupId] = { ...group, workers: uniqueWorkers };
                    }
                }
            }

            workerGroups = cleanedWorkerGroups;
            console.log('[WorkerGroups] 清理后的分组:', Object.keys(workerGroups));
            syncGroupsToServer();
        }

        const defaultGroup = workerGroups['default'];
        if (!defaultGroup) {
            createDefaultGroup();
            return;
        }

        let addedToDefault = [];
        for (const workerId of Object.keys(workersData)) {
            if (!allGroupedWorkers.has(workerId)) {
                defaultGroup.workers.push(workerId);
                addedToDefault.push(workerId);
            }
        }

        if (addedToDefault.length > 0) {
            console.log('[WorkerGroups] 添加 workers 到默认分组:', addedToDefault);
            syncGroupsToServer();
        }

        renderWorkerGroups();
        updateWorkersTable(workers);
    } catch (error) {
        console.error('Error refreshing workers:', error);
    }
}

async function loadInitialData() {
    try {
        const statusResponse = await fetch('/api/status');
        const status = await statusResponse.json();
        updateDashboard(status);

        const workersResponse = await fetch('/api/workers');
        const workers = await workersResponse.json();
        updateWorkersTable(workers);

        const healthResponse = await fetch('/api/health');
        const health = await healthResponse.json();
        updateHealthData(health);

        const stateResponse = await fetch('/api/state');
        const state = await stateResponse.json();
        updateStateSummary(state);

        await checkServerRestart();

        if (refreshWorkersTimer) {
            clearInterval(refreshWorkersTimer);
        }
        refreshWorkersTimer = setInterval(refreshWorkers, 2000);
    } catch (error) {
        console.error('Error loading initial data:', error);
    }
}

window.addEventListener('beforeunload', () => {
    if (refreshWorkersTimer) {
        clearInterval(refreshWorkersTimer);
        refreshWorkersTimer = null;
    }
    if (cleanupMemoryTimer) {
        clearInterval(cleanupMemoryTimer);
        cleanupMemoryTimer = null;
    }
    socket.disconnect();
});

function cleanupMemory() {
    const currentWorkerIds = new Set(Object.keys(workersData));

    for (const workerId of Object.keys(allLogs)) {
        if (!currentWorkerIds.has(workerId)) {
            delete allLogs[workerId];
        } else if (allLogs[workerId].length > MAX_LOGS) {
            allLogs[workerId] = allLogs[workerId].slice(0, MAX_LOGS);
        }
    }

    for (const workerId of Object.keys(healthData)) {
        if (!currentWorkerIds.has(workerId)) {
            delete healthData[workerId];
        }
    }

    for (const groupId of Object.keys(workerGroups)) {
        const group = workerGroups[groupId];
        if (group && group.workers) {
            group.workers = group.workers.filter(id => workersData[id]);
        }
    }

    if (filteredLogs.length > MAX_LOGS) {
        filteredLogs = filteredLogs.slice(0, MAX_LOGS);
    }

    const totalLogCount = Object.values(allLogs).reduce((sum, logs) => sum + logs.length, 0);
    console.log(`[Memory] Cleanup complete. Workers: ${currentWorkerIds.size}, Total logs: ${totalLogCount}`);
}

if (cleanupMemoryTimer) {
    clearInterval(cleanupMemoryTimer);
}
cleanupMemoryTimer = setInterval(cleanupMemory, 30000);

function renderWorkerGroups() {
    const container = document.getElementById('worker-groups');
    if (!container) return;

    // 如果配置弹窗打开，暂停刷新分组UI
    if (groupConfigModalOpen) {
        return;
    }
    
    const addCard = container.querySelector('.add-group-card');
    
    container.innerHTML = '';
    
    for (const [groupId, group] of Object.entries(workerGroups)) {
        const groupEl = createGroupElement(group, groupId);
        container.appendChild(groupEl);
    }
    
    container.appendChild(addCard);
    initSortables();
    updateGroupStats();
}

function createGroupElement(group, groupId) {
    const effectiveGroupId = group.id || groupId || 'default';
    const validWorkers = group.workers.filter(id => workersData[id]);
    const removedWorkers = group.workers.length - validWorkers.length;
    
    const runningCount = validWorkers.filter(id => {
        const worker = workersData[id];
        return worker && worker.status === 'running';
    }).length;
    
    const notRunningCount = validWorkers.filter(id => {
        const worker = workersData[id];
        return worker && worker.status !== 'running';
    }).length;
    
    const allSelected = validWorkers.length > 0 && validWorkers.every(id => selectedWorkers.has(id));
    
    const groupEl = document.createElement('div');
    groupEl.className = 'worker-group';
    groupEl.dataset.groupId = effectiveGroupId;
    
    const workerCountText = validWorkers.length + (removedWorkers > 0 ? ` (+${removedWorkers} 已移除)` : '');
    
    const sortedWorkers = [...validWorkers].sort((a, b) => {
        const workerA = workersData[a];
        const workerB = workersData[b];
        const isNotRunningA = workerA && workerA.status !== 'running';
        const isNotRunningB = workerB && workerB.status !== 'running';
        
        if (isNotRunningA && !isNotRunningB) return -1;
        if (!isNotRunningA && isNotRunningB) return 1;
        return 0;
    });
    
    groupEl.innerHTML = `
        <div class="worker-group-header">
            <div class="worker-group-title">
                <i class="fas fa-th-large"></i>
                <input type="text" value="${escapeHtml(group.name)}" 
                    onchange="renameGroup('${effectiveGroupId}', this.value)"
                    onclick="event.stopPropagation()">
            </div>
            <div class="worker-group-actions">
                <button class="btn btn-sm btn-light" onclick="showGroupConfigModal('${effectiveGroupId}')" title="PyTorch配置">
                    <i class="fas fa-cog"></i>
                </button>
                <button class="btn btn-sm btn-light" onclick="startGroupWorkers('${effectiveGroupId}')" title="批量启动">
                    <i class="fas fa-play"></i>
                </button>
                <button class="btn btn-sm btn-light" onclick="restartGroupWorkers('${effectiveGroupId}')" title="批量重启">
                    <i class="fas fa-redo"></i>
                </button>
                <button class="btn btn-sm btn-light" onclick="stopGroupWorkers('${effectiveGroupId}')" title="批量停止">
                    <i class="fas fa-stop"></i>
                </button>
                <button class="btn btn-sm btn-light" onclick="deleteGroup('${effectiveGroupId}')" title="删除分组">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
        <div class="worker-group-stats">
            <span><i class="fas fa-server"></i> ${workerCountText} 个</span>
            <span class="running"><i class="fas fa-check-circle"></i> ${runningCount}</span>
            <span class="not-running"><i class="fas fa-pause-circle"></i> ${notRunningCount}</span>
        </div>
        <div class="group-select-all">
            <div class="select-all-buttons">
                <button class="btn btn-sm btn-outline-primary" onclick="selectAllInGroup('${effectiveGroupId}', 'all')">
                    <i class="fas fa-check-square me-1"></i>全选
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="selectAllInGroup('${effectiveGroupId}', 'none')">
                    <i class="fas fa-square me-1"></i>取消
                </button>
                <button class="btn btn-sm btn-outline-warning" onclick="selectAllInGroup('${effectiveGroupId}', 'not-running')">
                    <i class="fas fa-pause-circle me-1"></i>暂停
                </button>
            </div>
        </div>
        <div class="worker-list" data-group-id="${effectiveGroupId}">
            ${sortedWorkers.map(workerId => createWorkerItem(workerId, effectiveGroupId)).join('')}
        </div>
        <div class="group-bulk-actions">
            <button class="btn btn-sm btn-info" onclick="showMoveGroupModal()" ${selectedWorkers.size === 0 ? 'disabled' : ''}>
                <i class="fas fa-exchange-alt me-1"></i>移动到
            </button>
            <button class="btn btn-sm btn-success" onclick="startGroupWorkers('${effectiveGroupId}')">
                <i class="fas fa-play me-1"></i>启动
            </button>
            <button class="btn btn-sm btn-primary" onclick="restartGroupWorkers('${effectiveGroupId}')">
                <i class="fas fa-redo me-1"></i>重启
            </button>
            <button class="btn btn-sm btn-danger" onclick="stopGroupWorkers('${effectiveGroupId}')">
                <i class="fas fa-stop me-1"></i>停止
            </button>
        </div>
    `;
    
    return groupEl;
}

function createWorkerItem(workerId, groupId) {
    const worker = workersData[workerId] || { status: 'unknown', pid: null, restart_count: 0 };
    const statusClass = getStatusClass(worker.status);
    const statusText = getStatusText(worker.status);
    const isSelected = selectedWorkers.has(workerId) ? 'checked' : '';
    const selectedClass = selectedWorkers.has(workerId) ? 'selected' : '';
    const isNotRunning = worker.status !== 'running';
    const notRunningClass = isNotRunning ? 'not-running' : '';
    
    return `
        <div class="worker-item ${selectedClass} ${notRunningClass}" data-worker-id="${workerId}" data-group-id="${groupId}">
            <input type="checkbox" class="worker-item-checkbox" 
                ${isSelected} onchange="toggleWorkerSelect('${workerId}')">
            <div class="worker-item-info">
                <span class="worker-item-id">${workerId}</span>
                <div class="worker-item-badges">
                    <span class="badge ${statusClass}">${statusText}</span>
                </div>
            </div>
            <div class="worker-item-actions">
                <button class="btn btn-sm btn-secondary" onclick="viewWorkerEnv('${workerId}')" title="环境变量">
                    <i class="fas fa-envira"></i>
                </button>
                <button class="btn btn-sm btn-info" onclick="viewWorkerLogs('${workerId}')" title="日志">
                    <i class="fas fa-file-alt"></i>
                </button>
                <button class="btn btn-sm btn-success" onclick="startWorker('${workerId}')" 
                    ${worker.status === 'running' ? 'disabled' : ''} title="启动">
                    <i class="fas fa-play"></i>
                </button>
                <button class="btn btn-sm btn-primary" onclick="restartWorker('${workerId}')" 
                    ${worker.status === 'stopped' ? 'disabled' : ''} title="重启">
                    <i class="fas fa-redo"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="stopWorker('${workerId}')" 
                    ${worker.status === 'running' ? '' : 'disabled'} title="停止">
                    <i class="fas fa-stop"></i>
                </button>
            </div>
        </div>
    `;
}

function toggleWorkerSelect(workerId) {
    if (selectedWorkers.has(workerId)) {
        selectedWorkers.delete(workerId);
    } else {
        selectedWorkers.add(workerId);
    }
    renderWorkerGroups();
}

function selectAllInGroup(groupId, mode = 'all') {
    const group = workerGroups[groupId];
    if (!group) return;
    
    const allInGroup = group.workers.filter(id => workersData[id]);
    
    if (mode === 'all') {
        allInGroup.forEach(id => selectedWorkers.add(id));
    } else if (mode === 'none') {
        allInGroup.forEach(id => selectedWorkers.delete(id));
    } else if (mode === 'not-running') {
        allInGroup.forEach(id => {
            const worker = workersData[id];
            if (worker && worker.status !== 'running') {
                selectedWorkers.add(id);
            } else {
                selectedWorkers.delete(id);
            }
        });
    }
    
    renderWorkerGroups();
}

function clearSelection() {
    selectedWorkers.clear();
    renderWorkerGroups();
}

function updateWorkerItem(workerId, data) {
    const item = document.querySelector(`.worker-item[data-worker-id="${workerId}"]`);
    if (!item) return;
    
    const statusClass = getStatusClass(data.status || 'unknown');
    const statusText = getStatusText(data.status || 'unknown');
    
    const actions = item.querySelector('.worker-item-actions');
    actions.innerHTML = `
        <button class="btn btn-sm btn-info" onclick="viewWorkerLogs('${workerId}')" title="日志">
            <i class="fas fa-file-alt"></i>
        </button>
        <button class="btn btn-sm btn-success" onclick="startWorker('${workerId}')" 
            ${data.status === 'running' ? 'disabled' : ''} title="启动">
            <i class="fas fa-play"></i>
        </button>
        <button class="btn btn-sm btn-primary" onclick="restartWorker('${workerId}')" 
            ${data.status === 'stopped' ? 'disabled' : ''} title="重启">
            <i class="fas fa-redo"></i>
        </button>
        <button class="btn btn-sm btn-danger" onclick="stopWorker('${workerId}')" 
            ${data.status === 'running' ? '' : 'disabled'} title="停止">
            <i class="fas fa-stop"></i>
        </button>
    `;
    
    const badges = item.querySelector('.worker-item-badges');
    if (badges) {
        badges.innerHTML = `<span class="badge ${statusClass}">${statusText}</span>`;
    }
}

function updateGroupStats() {
    for (const [groupId, group] of Object.entries(workerGroups)) {
        const groupEl = document.querySelector(`.worker-group[data-group-id="${groupId}"]`);
        if (!groupEl) continue;
        
        const validWorkers = group.workers.filter(id => workersData[id]);
        const removedWorkers = group.workers.length - validWorkers.length;
        
        const runningCount = validWorkers.filter(id => {
            const worker = workersData[id];
            return worker && worker.status === 'running';
        }).length;
        
        const notRunningCount = validWorkers.filter(id => {
            const worker = workersData[id];
            return worker && worker.status !== 'running';
        }).length;
        
        const statsEl = groupEl.querySelector('.worker-group-stats');
        const workerCountText = validWorkers.length + (removedWorkers > 0 ? ` (+${removedWorkers} 已移除)` : '');
        statsEl.innerHTML = `
            <span><i class="fas fa-server"></i> ${workerCountText} 个</span>
            <span class="running"><i class="fas fa-check-circle"></i> ${runningCount}</span>
            <span class="not-running"><i class="fas fa-pause-circle"></i> ${notRunningCount}</span>
        `;
    }
}

function showMoveGroupModal() {
    if (selectedWorkers.size === 0) {
        alert('请先选择要移动的 Worker');
        return;
    }
    
    const listEl = document.getElementById('move-group-list');
    listEl.innerHTML = '';
    
    for (const [groupId, group] of Object.entries(workerGroups)) {
        const item = document.createElement('button');
        item.className = 'list-group-item list-group-item-action';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span><i class="fas fa-th-large me-2"></i>${escapeHtml(group.name)}</span>
                <span class="badge bg-secondary">${group.workers.length} 个</span>
            </div>
        `;
        item.onclick = () => moveWorkersToGroup(groupId);
        listEl.appendChild(item);
    }

    moveGroupModal.show();
}

function closeMoveGroupModal() {
    moveGroupModal.hide();
}

function moveWorkersToGroup(targetGroupId) {
    const targetGroup = workerGroups[targetGroupId];
    if (!targetGroup) return;

    const movedWorkers = Array.from(selectedWorkers);

    for (const workerId of movedWorkers) {
        // 从其他分组移除
        for (const [groupId, group] of Object.entries(workerGroups)) {
            const idx = group.workers.indexOf(workerId);
            if (idx > -1) {
                group.workers.splice(idx, 1);
                break;
            }
        }
        // 添加到目标分组（如果不存在）
        if (!targetGroup.workers.includes(workerId)) {
            targetGroup.workers.push(workerId);
        }
    }

    selectedWorkers.clear();
    renderWorkerGroups();
    if (!groupConfigModalOpen) {
        saveWorkerGroupsConfig();
    }
    closeMoveGroupModal();
}

function initSortables() {
}

function saveWorkerGroupsConfig() {
    localStorage.setItem('procguard_groups', JSON.stringify(workerGroups));
    console.log('[WorkerGroups] 保存分组配置到本地缓存');
    syncGroupsToServer();
}

async function syncGroupsToServer() {
    try {
        const response = await fetch('/api/groups/sync', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                groups: workerGroups
            })
        });
        
        const result = await response.json();
        if (result.success) {
            console.log('分组信息已同步到服务器');
        } else {
            console.error('同步分组到服务器失败:', result.error);
        }
    } catch (error) {
        console.error('同步分组到服务器失败:', error);
    }
}

function loadGroupConfig() {
    const saved = localStorage.getItem('procguard_groups');
    if (saved) {
        try {
            workerGroups = JSON.parse(saved);
            console.log('[WorkerGroups] 从本地缓存加载分组配置');
            return true;
        } catch (e) {
            console.error('Error loading group config:', e);
        }
    }
    return false;
}

function showCreateGroupModal() {
    document.getElementById('group-name-input').value = '';
    document.getElementById('group-description').value = '';
    createGroupModal.show();
}

function closeCreateGroupModal() {
    createGroupModal.hide();
}

function createGroup() {
    const name = document.getElementById('group-name-input').value.trim();
    
    if (!name) {
        alert('请输入分组名称');
        return;
    }
    
    const groupId = 'group_' + Date.now();
    
    workerGroups[groupId] = {
        id: groupId,
        name: name,
        workers: []
    };
    
    renderWorkerGroups();
    saveWorkerGroupsConfig();
    closeCreateGroupModal();
}

function renameGroup(groupId, newName) {
    if (!newName.trim()) return;
    workerGroups[groupId].name = newName.trim();
    saveWorkerGroupsConfig();
}

function deleteGroup(groupId) {
    if (groupId === 'default') {
        alert('默认分组不能删除');
        return;
    }
    
    if (!confirm(`确定要删除分组 "${workerGroups[groupId].name}" 吗？组内的 Worker 将移到默认分组。`)) {
        return;
    }
    
    const workers = workerGroups[groupId].workers;
    const defaultGroup = workerGroups['default'];
    for (const workerId of workers) {
        if (!defaultGroup.workers.includes(workerId)) {
            defaultGroup.workers.push(workerId);
        }
    }
    delete workerGroups[groupId];
    
    renderWorkerGroups();
    saveWorkerGroupsConfig();
}

async function startGroupWorkers(groupId) {
    const workers = workerGroups[groupId].workers;
    if (workers.length === 0) {
        alert('该分组没有 Worker');
        return;
    }

    const runningWorkers = workers.filter(w => workersData[w] && workersData[w].status === 'running');
    const stoppedWorkers = workers.filter(w => !workersData[w] || workersData[w].status !== 'running');

    if (stoppedWorkers.length === 0) {
        alert('所有 Worker 已经在运行中');
        return;
    }

    let confirmMessage = `确定要启动分组 "${workerGroups[groupId].name}" 中的 ${stoppedWorkers.length} 个 Worker 吗？`;
    if (runningWorkers.length > 0) {
        confirmMessage += `\n\n注意：有 ${runningWorkers.length} 个 Worker 已经在运行中，将被跳过。`;
    }

    if (!confirm(confirmMessage)) {
        return;
    }

    let success = 0;
    let failed = 0;
    let skipped = 0;

    for (const workerId of workers) {
        if (workersData[workerId] && workersData[workerId].status === 'running') {
            skipped++;
            continue;
        }

        try {
            const response = await fetch(`/api/workers/${workerId}/start`, {
                method: 'POST'
            });
            const result = await response.json();
            if (result.success) {
                success++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Error starting worker:', workerId, error);
            failed++;
        }
    }

    alert(`启动完成: 成功 ${success} 个, 失败 ${failed} 个${skipped > 0 ? `, 跳过 ${skipped} 个` : ''}`);
}

async function restartGroupWorkers(groupId) {
    const workers = workerGroups[groupId].workers;
    if (workers.length === 0) {
        alert('该分组没有 Worker');
        return;
    }
    
    if (!confirm(`确定要重启分组 "${workerGroups[groupId].name}" 中的 ${workers.length} 个 Worker 吗？`)) {
        return;
    }
    
    let success = 0;
    let failed = 0;
    
    for (const workerId of workers) {
        try {
            const response = await fetch(`/api/workers/${workerId}/restart`, {
                method: 'POST'
            });
            const result = await response.json();
            if (result.success) {
                success++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Error restarting worker:', error);
            failed++;
        }
    }
    
    alert(`重启完成: 成功 ${success} 个, 失败 ${failed} 个`);
}

async function stopGroupWorkers(groupId) {
    const workers = workerGroups[groupId].workers;
    if (workers.length === 0) {
        alert('该分组没有 Worker');
        return;
    }
    
    if (!confirm(`确定要停止分组 "${workerGroups[groupId].name}" 中的 ${workers.length} 个 Worker 吗？`)) {
        return;
    }
    
    let success = 0;
    let failed = 0;
    
    for (const workerId of workers) {
        try {
            const response = await fetch(`/api/workers/${workerId}/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force: true })
            });
            const result = await response.json();
            if (result.success) {
                success++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Error stopping worker:', error);
            failed++;
        }
    }
    
    alert(`停止完成: 成功 ${success} 个, 失败 ${failed} 个`);
}

function updateConnectionStatus(connected) {
    const statusIcon = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');
    
    if (connected) {
        statusIcon.className = 'fas fa-circle text-success me-2';
        statusText.textContent = '已连接';
    } else {
        statusIcon.className = 'fas fa-circle text-danger me-2';
        statusText.textContent = '已断开';
    }
}

function updateDashboard(data) {
    if (data.state) {
        updateStateSummary(data.state);
    }
}

function updateStateSummary(state) {
    document.getElementById('total-workers').textContent = state.total_workers || 0;
}

function updateRecoveryStats(stats) {
}

function updateWorkersTable(workers) {
    const tbody = document.getElementById('workers-tbody');
    tbody.innerHTML = '';
    
    for (const [workerId, worker] of Object.entries(workers)) {
        workersData[workerId] = worker;
        addWorkerRow(workerId, worker);
    }
}

function addWorkerRow(workerId, worker) {
    const tbody = document.getElementById('workers-tbody');
    const existingRow = document.getElementById(`worker-row-${workerId}`);

    if (existingRow) {
        const statusCell = existingRow.querySelector('td:nth-child(2)');
        const pidCell = existingRow.querySelector('td:nth-child(3)');
        const restartCell = existingRow.querySelector('td:nth-child(4)');

        const statusClass = getStatusClass(worker.status);
        const statusText = getStatusText(worker.status);

        if (statusCell) {
            statusCell.innerHTML = `<span class="badge ${statusClass}">${statusText}</span>`;
        }
        if (pidCell) {
            pidCell.innerHTML = `<code>${worker.pid || 'N/A'}</code>`;
        }
        if (restartCell) {
            restartCell.textContent = worker.restart_count || 0;
        }

        const buttons = existingRow.querySelectorAll('button');
        if (buttons.length >= 4) {
            buttons[1].disabled = worker.status === 'running';
            buttons[2].disabled = worker.status === 'stopped';
            buttons[3].disabled = worker.status !== 'running';
        }

        workersData[workerId] = worker;
        return;
    }

    const statusClass = getStatusClass(worker.status);
    const statusText = getStatusText(worker.status);

    const row = document.createElement('tr');
    row.id = `worker-row-${workerId}`;
    row.innerHTML = `
        <td><strong>${workerId}</strong></td>
        <td><span class="badge ${statusClass}">${statusText}</span></td>
        <td><code>${worker.pid || 'N/A'}</code></td>
        <td>${worker.restart_count || 0}</td>
        <td>
            <button class="btn btn-sm btn-info me-1" onclick="viewWorkerLogs('${workerId}')" title="查看日志">
                <i class="fas fa-file-alt"></i>
            </button>
            <button class="btn btn-sm btn-success me-1" onclick="startWorker('${workerId}')" ${worker.status === 'running' ? 'disabled' : ''}>
                <i class="fas fa-play"></i>
            </button>
            <button class="btn btn-sm btn-primary me-1" onclick="restartWorker('${workerId}')" ${worker.status === 'stopped' ? 'disabled' : ''}>
                <i class="fas fa-redo"></i>
            </button>
            <button class="btn btn-sm btn-danger" onclick="stopWorker('${workerId}')" ${worker.status === 'running' ? '' : 'disabled'}>
                <i class="fas fa-stop"></i>
            </button>
        </td>
    `;

    tbody.appendChild(row);
    workersData[workerId] = worker;
}

function updateWorkerRow(workerId, data) {
    if (workersData[workerId]) {
        workersData[workerId] = { ...workersData[workerId], ...data };
        addWorkerRow(workerId, workersData[workerId]);
    }
}

function removeWorkerRow(workerId) {
    const row = document.getElementById(`worker-row-${workerId}`);
    if (row) {
        row.remove();
        console.log(`[Worker] 已移除 worker 行: ${workerId}`);
    }
}

function getStatusClass(status) {
    const statusClassMap = {
        'running': 'bg-success',
        'stopped': 'bg-secondary',
        'failed': 'bg-danger',
        'unknown': 'bg-warning'
    };
    return statusClassMap[status.toLowerCase()] || 'bg-secondary';
}

function getStatusText(status) {
    const statusTextMap = {
        'running': '运行中',
        'stopped': '已停止',
        'failed': '失败',
        'unknown': '未知'
    };
    return statusTextMap[status.toLowerCase()] || status;
}

function updateHealthData(health) {
    healthData = health;
    
    for (const [workerId, report] of Object.entries(health)) {
        const row = document.getElementById(`worker-row-${workerId}`);
        if (row) {
            const statusClass = getStatusClass(report.status);
            const statusText = getStatusText(report.status);
            
            const statusCell = row.querySelector('td:nth-child(2)');
            if (statusCell) {
                statusCell.innerHTML = `<span class="badge ${statusClass}">${statusText}</span>`;
            }
        }
    }
}

function updateRecoveryHistory(history) {
}

function addRecoveryHistory(data) {
}

async function restartWorker(workerId) {
    if (!confirm(`确定要重启 Worker "${workerId}" 吗？`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/workers/${workerId}/restart`, {
            method: 'POST'
        });
        const result = await response.json();
        
        if (!result.success) {
            alert(`Worker ${workerId} 重启失败`);
        }
    } catch (error) {
        console.error('Error restarting worker:', error);
        alert(`重启 Worker ${workerId} 时出错: ${error.message}`);
    }
}

async function stopWorker(workerId) {
    if (!confirm(`确定要停止 Worker "${workerId}" 吗？`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/workers/${workerId}/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ force: true })
        });
        const result = await response.json();
        
        if (!result.success) {
            alert(`Worker ${workerId} 停止失败`);
        }
    } catch (error) {
        console.error('Error stopping worker:', error);
        alert(`停止 Worker ${workerId} 时出错: ${error.message}`);
    }
}

async function startWorker(workerId) {
    try {
        const response = await fetch(`/api/workers/${workerId}/start`, {
            method: 'POST'
        });
        const result = await response.json();
        
        if (!result.success) {
            alert(`Worker ${workerId} 启动失败`);
        }
    } catch (error) {
        console.error('Error starting worker:', error);
        alert(`启动 Worker ${workerId} 时出错: ${error.message}`);
    }
}

let currentWorkerId = null;
let logPanelExpanded = false;
let autoScroll = true;
let allLogs = {};
let filteredLogs = [];

window.debugLogs = function() {
    console.log('=== All Received Logs ===');
    for (const [workerId, logs] of Object.entries(allLogs)) {
        console.log(`${workerId}: ${logs.length} logs`);
        logs.slice(0, 3).forEach(log => {
            console.log(`  [${log.timestamp}] [${log.level}] ${log.message.substring(0, 100)}`);
        });
        if (logs.length > 3) {
            console.log(`  ... and ${logs.length - 3} more`);
        }
    }
    console.log('currentWorkerId:', currentWorkerId);
    console.log('=======================');
};

socket.on('log_message', (data) => {
    const workerId = data.worker_id;
    if (!workerId) {
        console.warn('Log message without worker_id:', data);
        return;
    }
    
    if (!allLogs[workerId]) {
        allLogs[workerId] = [];
    }
    
    const timestamp = new Date().toLocaleTimeString();
    allLogs[workerId].unshift({
        timestamp,
        message: data.message,
        level: data.level || 'info'
    });
    
    if (allLogs[workerId].length > MAX_LOGS) {
        allLogs[workerId].pop();
    }
    
    console.log(`Log received for ${workerId}:`, data.message.substring(0, 50));
    
    if (currentWorkerId === workerId) {
        updateLogDisplay();
    }
});

function addLog(message, level = 'info') {
    if (!currentWorkerId || !allLogs[currentWorkerId]) return;
    
    const timestamp = new Date().toLocaleTimeString();
    allLogs[currentWorkerId].unshift({ timestamp, message, level });
    
    if (allLogs[currentWorkerId].length > MAX_LOGS) {
        allLogs[currentWorkerId].pop();
    }
    
    updateLogDisplay();
}

function updateLogDisplay() {
    if (!currentWorkerId || !allLogs[currentWorkerId]) {
        document.getElementById('log-viewer-main').innerHTML = '<div class="log-placeholder">选择 Worker 查看日志</div>';
        return;
    }
    
    const filterText = document.getElementById('log-filter').value.toLowerCase();
    const viewer = document.getElementById('log-viewer-main');
    
    const workerLogs = allLogs[currentWorkerId] || [];
    filteredLogs = workerLogs.filter(log => 
        log.message.toLowerCase().includes(filterText)
    );
    
    if (filteredLogs.length === 0) {
        viewer.innerHTML = '<div class="log-placeholder">暂无日志</div>';
    } else {
        viewer.innerHTML = filteredLogs.map(log => `
            <div class="log-line">
                <span class="timestamp">[${log.timestamp}]</span>
                <span class="level-${log.level}">${escapeHtml(log.message)}</span>
            </div>
        `).join('');
    }
    
    const totalCount = Object.values(allLogs).reduce((sum, logs) => sum + logs.length, 0);
    document.getElementById('log-count').textContent = (allLogs[currentWorkerId] || []).length;
    document.getElementById('log-count-all').textContent = totalCount;
    
    if (autoScroll && filteredLogs.length > 0) {
        viewer.scrollTop = 0;
    }
}

function toggleLogPanel() {
    logPanelExpanded = !logPanelExpanded;
    const panel = document.getElementById('logPanel');
    panel.classList.toggle('expanded', logPanelExpanded);
}

function filterLogs() {
    updateLogDisplay();
}

function clearAllLogs() {
    if (currentWorkerId && allLogs[currentWorkerId]) {
        allLogs[currentWorkerId] = [];
    }
    filteredLogs = [];
    updateLogDisplay();
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const icon = document.getElementById('scroll-icon');
    icon.className = autoScroll ? 'fas fa-arrow-down' : 'fas fa-pause';
}

async function viewWorkerLogs(workerId) {
    currentWorkerId = workerId;
    
    document.getElementById('log-header').textContent = `Worker: ${workerId}`;
    
    if (!allLogs[workerId]) {
        allLogs[workerId] = [];
    }
    
    const viewer = document.getElementById('log-viewer');
    
    if (allLogs[workerId].length > 0) {
        displayLogs(allLogs[workerId].map(log => `[${log.timestamp}] [${log.level.toUpperCase()}] ${log.message}`));
    } else {
        viewer.innerHTML = '<div class="log-modal-placeholder">加载日志中...</div>';
    }
    
    logModal.show();
    
    try {
        const response = await fetch(`/api/workers/${workerId}/logs`);
        const data = await response.json();
        if (data.logs && data.logs.length > 0) {
            allLogs[workerId] = data.logs.map(log => {
                const match = log.match(/^\[(.*?)\] \[(.*?)\] (.*)$/);
                return match ? {
                    timestamp: match[1],
                    level: match[2].toLowerCase(),
                    message: match[3]
                } : { timestamp: '', level: 'info', message: log };
            });
            if (currentWorkerId === workerId) {
                displayLogs(data.logs);
            }
        } else if (allLogs[workerId].length === 0) {
            viewer.innerHTML = '<div class="log-modal-placeholder">暂无日志</div>';
        }
    } catch (error) {
        console.error('Error fetching logs:', error);
        if (allLogs[workerId].length === 0) {
            viewer.innerHTML = '<div class="log-modal-placeholder">加载日志失败</div>';
        }
    }
}

function displayLogs(logs) {
    const viewer = document.getElementById('log-viewer');
    
    if (!logs || logs.length === 0) {
        viewer.innerHTML = '<div class="log-modal-placeholder">暂无日志</div>';
        return;
    }
    
    viewer.innerHTML = logs.map(log => {
        return `<div class="log-modal-line">${escapeHtml(log)}</div>`;
    }).join('');
    
    viewer.scrollTop = viewer.scrollHeight;
}

async function refreshWorkerLogs() {
    if (!currentWorkerId) return;
    
    try {
        const response = await fetch(`/api/workers/${currentWorkerId}/logs`);
        const data = await response.json();
        if (data.logs) {
            allLogs[currentWorkerId] = data.logs.map(log => {
                const match = log.match(/^\[(.*?)\] \[(.*?)\] (.*)$/);
                return match ? {
                    timestamp: match[1],
                    level: match[2].toLowerCase(),
                    message: match[3]
                } : { timestamp: '', level: 'info', message: log };
            });
            displayLogs(data.logs);
        }
    } catch (error) {
        console.error('Error refreshing logs:', error);
    }
}

async function clearWorkerLogs() {
    if (!currentWorkerId) return;
    
    if (!confirm(`确定要清空 Worker "${currentWorkerId}" 的日志吗？`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/workers/${currentWorkerId}/logs`, {
            method: 'DELETE'
        });
        const data = await response.json();
        
        if (data.success) {
            allLogs[currentWorkerId] = [];
            displayLogs([]);
        }
    } catch (error) {
        console.error('Error clearing logs:', error);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function switchLogTab(tabName) {
    document.querySelectorAll('.log-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`.log-tab[data-tab="${tabName}"]`).classList.add('active');
    updateLogDisplay();
}

let pytorchConfigModal;
let workerEnvModal;
let currentWorkerEnv = {};
let currentWorkerIdForEnv = null;

function initWorkerEnvModal() {
    const modalEl = document.getElementById('workerEnvModal');
    if (modalEl) {
        workerEnvModal = new bootstrap.Modal(modalEl);
    }
}

async function viewWorkerEnv(workerId) {
    currentWorkerIdForEnv = workerId;
    currentWorkerEnv = {};

    const tbody = document.getElementById('worker-env-tbody');
    tbody.innerHTML = '<tr><td colspan="2" class="text-center">加载中...</td></tr>';

    document.getElementById('worker-env-title').textContent = `Worker 环境变量: ${workerId}`;

    try {
        const response = await fetch(`/api/pytorch/env?worker_id=${workerId}`);

        if (response.ok) {
            const envVars = await response.json();
            currentWorkerEnv = envVars;
            renderWorkerEnvTable(envVars);
        } else {
            tbody.innerHTML = '<tr><td colspan="2" class="text-center text-danger">获取环境变量失败</td></tr>';
        }
    } catch (error) {
        console.error('Error fetching worker env:', error);
        tbody.innerHTML = '<tr><td colspan="2" class="text-center text-danger">获取环境变量失败</td></tr>';
    }

    if (!workerEnvModal) {
        initWorkerEnvModal();
    }
    workerEnvModal.show();
}

function renderWorkerEnvTable(envVars) {
    const tbody = document.getElementById('worker-env-tbody');
    tbody.innerHTML = '';
    
    const pytorchVars = [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
        'CUDA_VISIBLE_DEVICES', 'NCCL_SOCKET_IFNAME', 
        'TORCH_DISTRIBUTED_BACKEND', 'CUDA_LAUNCH_BLOCKING'
    ];
    
    const sortedKeys = Object.keys(envVars).sort((a, b) => {
        const aIsPyTorch = pytorchVars.includes(a);
        const bIsPyTorch = pytorchVars.includes(b);
        if (aIsPyTorch && !bIsPyTorch) return -1;
        if (!aIsPyTorch && bIsPyTorch) return 1;
        return a.localeCompare(b);
    });
    
    if (sortedKeys.length === 0) {
        tbody.innerHTML = '<tr><td colspan="2" class="text-center text-muted">暂无环境变量配置</td></tr>';
        return;
    }
    
    for (const key of sortedKeys) {
        const value = envVars[key];
        const isPyTorch = pytorchVars.includes(key);
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${escapeHtml(key)}</strong>${isPyTorch ? '<span class="badge bg-primary ms-1">PyTorch</span>' : ''}</td>
            <td><code>${escapeHtml(value)}</code></td>
        `;
        tbody.appendChild(row);
    }
}

function copyWorkerEnv() {
    if (Object.keys(currentWorkerEnv).length === 0) {
        alert('没有可复制的环境变量');
        return;
    }
    
    const lines = Object.entries(currentWorkerEnv).map(([key, value]) => `${key}=${value}`);
    const text = lines.join('\n');
    
    navigator.clipboard.writeText(text).then(() => {
        alert('环境变量已复制到剪贴板');
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('复制失败，请手动选择复制');
    });
}

function exportWorkerEnv() {
    if (Object.keys(currentWorkerEnv).length === 0) {
        alert('没有可导出的环境变量');
        return;
    }
    
    const lines = Object.entries(currentWorkerEnv).map(([key, value]) => `${key}=${value}`);
    const text = lines.join('\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pytorch_env_${currentWorkerIdForEnv || 'worker'}.sh`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showGroupConfigModal(groupId) {
    // 确保groupId有效
    if (!groupId || groupId === 'undefined' || groupId === 'null') {
        groupId = 'default';
    }
    
    // 确保workerGroups存在
    if (!workerGroups || Object.keys(workerGroups).length === 0) {
        createDefaultGroup();
    }
    
    let group = workerGroups[groupId];
    
    // 如果分组不存在，尝试自动恢复
    if (!group) {
        // 重新从localStorage加载
        const saved = localStorage.getItem('procguard_groups');
        if (saved) {
            try {
                const savedGroups = JSON.parse(saved);
                workerGroups = savedGroups;
                group = workerGroups[groupId];
            } catch (e) {
                console.error('Error reloading groups:', e);
            }
        }
        
        // 如果还是不存在，遍历查找
        if (!group) {
            for (const [gid, g] of Object.entries(workerGroups)) {
                if (g && g.name === groupId) {
                    groupId = gid;
                    group = g;
                    break;
                }
            }
        }
        
        // 如果还是不存在，使用默认分组
        if (!group) {
            group = workerGroups['default'];
            if (group) {
                groupId = 'default';
            }
        }
    }
    
    // 最终检查
    if (!group) {
        return;
    }
    
    const pytorchConfig = group.pytorch_config || {};
    document.getElementById('config-group-id').value = groupId || 'default';
    document.getElementById('config-group-name').value = group.name || groupId || '默认分组';
    document.getElementById('config-master-addr').value = pytorchConfig.master_addr || '';
    document.getElementById('config-world-size').value = pytorchConfig.world_size || '';
    document.getElementById('config-master-port').value = pytorchConfig.master_port || 29500;
    document.getElementById('config-backend').value = pytorchConfig.backend || 'nccl';
    
    groupConfigModalOpen = true;
    groupConfigModal.show();
}

async function saveGroupConfig(isManualSave = false) {
    // 自动保存（刷新时调用）只有在配置弹窗关闭时才执行
    // 手动保存（用户点击按钮）总是执行
    if (!isManualSave && groupConfigModalOpen) {
        return;
    }
    
    let groupId = document.getElementById('config-group-id').value;
    const masterAddr = document.getElementById('config-master-addr').value.trim();
    const worldSize = document.getElementById('config-world-size').value.trim();
    const masterPort = document.getElementById('config-master-port').value;
    const backend = document.getElementById('config-backend').value;
    
    // 确保workerGroups存在
    if (!workerGroups || Object.keys(workerGroups).length === 0) {
        createDefaultGroup();
    }
    
    let group = workerGroups[groupId];
    if (!group) {
        // 尝试从localStorage恢复
        const saved = localStorage.getItem('procguard_groups');
        if (saved) {
            try {
                workerGroups = JSON.parse(saved);
                group = workerGroups[groupId];
            } catch (e) {
                console.error('Error reloading groups:', e);
            }
        }
        
        // 如果还是不存在，尝试使用默认分组
        if (!group) {
            group = workerGroups['default'];
            if (group) {
                groupId = 'default';
            }
        }
        
        if (!group) {
            alert('分组不存在且无法恢复');
            return;
        }
    }
    
    // 如果groupId仍然为空，使用默认
    if (!groupId) {
        groupId = 'default';
        group = workerGroups['default'];
        if (!group) {
            alert('分组不存在');
            return;
        }
    }
    
    const pytorch_config = {
        master_addr: masterAddr || null,
        master_port: parseInt(masterPort) || 29500,
        world_size: worldSize ? parseInt(worldSize) : null,
        backend: backend
    };
    
    try {
        const response = await fetch(`/api/groups/${groupId}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: group.name || groupId,
                ...pytorch_config,
                workers: group.workers
            })
        });
        
        if (response.ok) {
            if (!group.pytorch_config) {
                group.pytorch_config = {};
            }
            Object.assign(group.pytorch_config, pytorch_config);
            groupConfigModalOpen = false;
            groupConfigModal.hide();
            renderWorkerGroups();
        } else {
            const data = await response.json();
            console.error('保存配置失败: ' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('Error saving group config:', error);
    }
}

function addGroupConfigButton() {
    const groupHeader = document.querySelector('.worker-group-header');
    if (groupHeader && !groupHeader.querySelector('.config-btn')) {
        const configBtn = document.createElement('button');
        configBtn.className = 'btn btn-sm btn-light config-btn';
        configBtn.title = 'PyTorch配置';
        configBtn.innerHTML = '<i class="fas fa-cog"></i>';
        configBtn.onclick = function(e) {
            e.stopPropagation();
            const groupEl = this.closest('.worker-group');
            if (groupEl) {
                showGroupConfigModal(groupEl.dataset.groupId);
            }
        };
        groupHeader.querySelector('.worker-group-actions').appendChild(configBtn);
    }
}

async function clearGroupCache() {
    if (!confirm('确定要清除分组缓存吗？这将删除所有本地分组配置并重新从服务器同步。')) {
        return;
    }
    
    try {
        const response = await fetch('/api/groups/cache', {
            method: 'DELETE'
        });
        
        const result = await response.json();
        if (result.success) {
            localStorage.removeItem('procguard_groups');
            workerGroups = {};
            renderWorkerGroups();
            alert('分组缓存已清除，页面将刷新...');
            location.reload();
        } else {
            alert('清除缓存失败: ' + (result.error || '未知错误'));
        }
    } catch (error) {
        console.error('Error clearing group cache:', error);
        alert('清除缓存时出错: ' + error.message);
    }
}
