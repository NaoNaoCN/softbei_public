// html-test/assets/api.js
// 统一 API 调用层，所有 HTML 页面通过此模块与后端通信

const API_BASE = window.location.origin;

export function getUserId() {
    return localStorage.getItem('user_id');
}

export function getToken() {
    return localStorage.getItem('access_token');
}

export function setAuth(user_id, access_token) {
    localStorage.setItem('user_id', user_id);
    localStorage.setItem('access_token', access_token);
}

export function clearAuth() {
    localStorage.removeItem('user_id');
    localStorage.removeItem('access_token');
    localStorage.removeItem('session_id');
    sessionStorage.removeItem('softbei_daily_shown');
}

export function isLoggedIn() {
    return !!localStorage.getItem('user_id');
}

export async function apiFetch(endpoint, options = {}) {
    const token = getToken();
    const headers = { ...(options.headers || {}) };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    const resp = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });
    if (resp.status === 401) {
        clearAuth();
        window.location.href = 'auth.html';
        return null;
    }
    if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`API ${resp.status}: ${body.slice(0, 200)}`);
    }
    return resp;
}

export async function authLogin(username, password) {
    try {
        const resp = await apiFetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        if (!resp) return null;
        const data = await resp.json();
        if (resp.ok) {
            setAuth(data.user_id, data.access_token);
        }
        return data;
    } catch (e) {
        console.error('登录请求失败:', e);
        return null;
    }
}

export async function authRegister(username, password, email = null) {
    try {
        const body = { username, password };
        if (email) body.email = email;
        const resp = await apiFetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return resp?.json();
    } catch (e) {
        console.error('注册请求失败:', e);
        return null;
    }
}

export async function getProfile(user_id) {
    const resp = await apiFetch(`/profile?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function updateProfile(user_id, data) {
    const resp = await apiFetch(`/profile?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function updateAccount(user_id, data) {
    const resp = await apiFetch(`/user/account?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function getUser(user_id) {
    const resp = await apiFetch(`/user?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

// 注销（硬删除）账号：需提供用户名 + 密码双重确认。成功后由调用方清理本地登录态。
export async function deleteAccount(user_id, username, password) {
    const resp = await apiFetch(`/user/account?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    return resp?.json();
}

export async function getResources(user_id, { resource_type, kp_id, skip, limit } = {}) {
    let url = `/resources?user_id=${encodeURIComponent(user_id)}`;
    if (resource_type) url += `&resource_type=${encodeURIComponent(resource_type)}`;
    if (kp_id) url += `&kp_id=${encodeURIComponent(kp_id)}`;
    if (skip !== undefined) url += `&skip=${skip}`;
    if (limit !== undefined) url += `&limit=${limit}`;
    const resp = await apiFetch(url);
    return resp?.json();
}

export async function getResourceStats(user_id) {
    const resp = await apiFetch(`/resources/stats?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function getResource(resource_id, user_id) {
    let url = `/resources/${resource_id}`;
    if (user_id) url += `?user_id=${encodeURIComponent(user_id)}`;
    const resp = await apiFetch(url);
    return resp?.json();
}


export async function deleteResource(resource_id, user_id) {
    const resp = await apiFetch(`/resources/${resource_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE'
    });
    return resp?.ok;
}

export async function startGeneration(user_id, kp_id, resource_type, num_questions = 4, question_type_counts = null) {
    const body = { kp_id, resource_type, num_questions };
    if (question_type_counts) body.question_type_counts = question_type_counts;
    const resp = await apiFetch(`/generate?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return resp?.json();
}

export async function getTaskStatus(task_id) {
    const resp = await apiFetch(`/generate/${task_id}/status`);
    return resp?.json();
}

export async function startBatchGeneration(userId, kpId, resourceTypes, numQuestions = 4, questionTypeCounts = null) {
    const body = { kp_id: kpId, resource_types: resourceTypes, num_questions: numQuestions };
    if (questionTypeCounts) body.question_type_counts = questionTypeCounts;
    const resp = await apiFetch(`/generate/batch?user_id=${encodeURIComponent(userId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return resp?.json();
}

export async function getBatchStatus(batchId) {
    const resp = await apiFetch(`/generate/batch/${batchId}/status`);
    return resp?.json();
}

export async function smartPlanResources(userId, kpId) {
    const resp = await apiFetch(`/generate/smart?user_id=${encodeURIComponent(userId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kp_id: kpId })
    });
    return resp?.json();
}

export async function getKgGraph(params = {}) {
    let url = '/kg/graph?';
    const searchParams = new URLSearchParams();
    if (params.root_id) searchParams.set('root_id', params.root_id);
    if (params.doc_id) searchParams.set('doc_id', params.doc_id);
    if (params.user_id) searchParams.set('user_id', params.user_id);
    if (params.depth) searchParams.set('depth', params.depth);
    url += searchParams.toString();
    const resp = await apiFetch(url);
    return resp?.json();
}

export async function getPathways(user_id) {
    const resp = await apiFetch(`/pathways?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function createPathway(user_id, name) {
    const resp = await apiFetch(`/pathways?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    return resp?.json();
}

export async function addPathwayItem(path_id, user_id, kp_id, order_index) {
    const resp = await apiFetch(`/pathways/${path_id}/items?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kp_id, order_index })
    });
    return resp?.json();
}

export async function markPathwayItemDone(path_id, item_id, user_id, is_completed = true) {
    const resp = await apiFetch(`/pathways/${path_id}/items/${item_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_completed })
    });
    return resp?.ok;
}

export async function deletePathway(path_id, user_id) {
    const resp = await apiFetch(`/pathways/${path_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE'
    });
    return resp?.ok;
}

export async function getChatSessions(user_id) {
    const resp = await apiFetch(`/chat/sessions?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function createChatSession(user_id) {
    const resp = await apiFetch(`/chat/sessions?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST'
    });
    return resp?.json();
}

export async function getChatMessages(session_id, user_id) {
    const resp = await apiFetch(`/chat/${session_id}/messages?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function sendChatMessage(session_id, user_id, message, signal) {
    const opts = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: message })
    };
    if (signal) opts.signal = signal;
    const resp = await apiFetch(
        `/chat/${session_id}?user_id=${encodeURIComponent(user_id)}&stream=false`,
        opts
    );
    return resp?.json();
}

export async function deleteChatSession(session_id, user_id) {
    const resp = await apiFetch(`/chat/sessions/${session_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE'
    });
    return resp?.ok;
}

export async function updateSessionTitle(session_id, user_id, title) {
    const resp = await apiFetch(`/chat/sessions/${session_id}/title?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
    });
    return resp?.ok;
}

export async function getQuizItems(resource_id, user_id) {
    let url = `/resources/${resource_id}/quiz`;
    if (user_id) url += `?user_id=${encodeURIComponent(user_id)}`;
    const resp = await apiFetch(url);
    return resp?.json();
}

export async function submitQuizAnswer(user_id, quiz_item_id, user_answer, duration_seconds = null) {
    const body = { quiz_item_id, user_answer };
    if (duration_seconds) body.duration_seconds = duration_seconds;
    const resp = await apiFetch(`/quiz/submit?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return resp?.json();
}

export async function getQuizAttempts(user_id, limit = 50) {
    const resp = await apiFetch(`/quiz/attempts?user_id=${encodeURIComponent(user_id)}&limit=${limit}`);
    return resp?.json();
}

export async function getDocuments(user_id, skip, limit) {
    let url = `/documents?user_id=${encodeURIComponent(user_id)}`;
    if (skip !== undefined) url += `&skip=${skip}`;
    if (limit !== undefined) url += `&limit=${limit}`;
    const resp = await apiFetch(url);
    return resp?.json();
}

export async function importDocumentAsync(user_id, file, title) {
    const formData = new FormData();
    formData.append('file', file);
    if (title) {
        console.log('[importDocumentAsync] appending title:', JSON.stringify(title));
        formData.append('title', title);
    }
    const headers = {};
    const token = getToken();
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    const resp = await fetch(`${API_BASE}/documents/import/async?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers,
        body: formData
    });
    return resp?.json();
}

export async function getImportStatus(task_id) {
    const resp = await apiFetch(`/documents/import/${task_id}/status`);
    return resp?.json();
}

export async function buildKg(doc_id, user_id) {
    const resp = await apiFetch('/kg/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id, user_id })
    });
    return resp?.json();
}

export async function getKgBuildStatus(task_id) {
    const resp = await apiFetch(`/kg/build/${task_id}/status`);
    return resp?.json();
}

export async function getKgBuildStatusByDoc(doc_id) {
    const resp = await apiFetch(`/kg/build/by-doc/${doc_id}/status`);
    return resp?.json();
}

export async function deleteDocument(doc_id, user_id) {
    const resp = await apiFetch(`/documents/${doc_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE'
    });
    return resp?.ok;
}

export async function postLearningRecord(user_id, data) {
    const resp = await apiFetch(`/records?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function getLearningRecords(user_id, { kp_id, limit, skip } = {}) {
    let url = `/records?user_id=${encodeURIComponent(user_id)}`;
    if (kp_id) url += `&kp_id=${encodeURIComponent(kp_id)}`;
    if (limit) url += `&limit=${limit}`;
    if (skip) url += `&skip=${skip}`;
    const resp = await apiFetch(url);
    return resp?.json();
}

export async function getLearningAnalytics(user_id) {
    const resp = await apiFetch(`/analytics/dashboard?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function sendStudyPlanEmail(plan_id, user_id) {
    const resp = await apiFetch(`/study-plan/email?plan_id=${plan_id}&user_id=${user_id}`, {
        method: 'POST'
    });
    return resp?.json();
}

export async function listStudyPlans(user_id) {
    const resp = await apiFetch(`/study-plans?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function getStudyPlan(plan_id, user_id) {
    const resp = await apiFetch(`/study-plans/${plan_id}?user_id=${encodeURIComponent(user_id)}`);
    return resp?.json();
}

export async function generateStudyPlan(user_id, data) {
    const resp = await apiFetch(`/study-plans/generate?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function updateStudyPlan(plan_id, user_id, data) {
    const resp = await apiFetch(`/study-plans/${plan_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function deleteStudyPlan(plan_id, user_id) {
    const resp = await apiFetch(`/study-plans/${plan_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'DELETE'
    });
    return resp?.ok;
}

export async function updateStudyPlanItem(plan_id, item_id, user_id, data) {
    const resp = await apiFetch(`/study-plans/${plan_id}/items/${item_id}?user_id=${encodeURIComponent(user_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return resp?.json();
}

export async function generateStudyPlanItemResources(plan_id, item_id, user_id, resource_types) {
    const resp = await apiFetch(`/study-plans/${plan_id}/items/${item_id}/generate-resource?user_id=${encodeURIComponent(user_id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resource_types })
    });
    return resp?.json();
}