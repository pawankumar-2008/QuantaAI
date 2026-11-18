import { API_BASE_URL } from "../config.js";

let csrfRequest;

async function fetchJson(path, options) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: "include",
    });
  } catch {
    throw new Error(
      `Could not connect to QuantaAI at ${API_BASE_URL}. Check that FastAPI is running.`,
    );
  }

  let data;
  try {
    data = response.status === 204 ? null : await response.json();
  } catch {
    throw new Error(
      `The backend returned an unreadable response (HTTP ${response.status}).`,
    );
  }

  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail
          .map((item) =>
            typeof item === "object" && item !== null
              ? item.msg
              : String(item),
          )
          .filter(Boolean)
          .join(" ")
      : typeof detail === "string"
        ? detail
        : `The backend rejected the request (HTTP ${response.status}).`;

    const error = new Error(message);
    error.status = response.status;
    throw error;
  }

  return data;
}

async function csrfHeaders() {
  if (!csrfRequest) {
    csrfRequest = fetchJson("/auth/csrf").finally(() => {
      csrfRequest = undefined;
    });
  }
  const { csrf_token: csrfToken } = await csrfRequest;
  return { "X-CSRF-Token": csrfToken };
}

export async function registerUser(account) {
  const headers = await csrfHeaders();
  return fetchJson("/auth/register", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(account),
  });
}

export async function loginUser(credentials) {
  const headers = await csrfHeaders();
  return fetchJson("/auth/login", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(credentials),
  });
}

export async function getCurrentUser() {
  return fetchJson("/auth/me");
}

export async function updateCurrentUser(profile) {
  const headers = await csrfHeaders();
  return fetchJson("/auth/me", {
    method: "PATCH",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(profile),
  });
}

export async function logoutUser() {
  const headers = await csrfHeaders();
  return fetchJson("/auth/logout", {
    method: "POST",
    headers,
  });
}

export function getBackendHealth() {
  return fetchJson("/health");
}

export async function sendChatMessage({
  message,
  agent,
  mode,
  tutorMode = "direct",
  conversationHistory = [],
  documentId,
  conversationId,
}) {
  const csrf = await csrfHeaders();
  const body = {
    message,
    agent,
    mode,
    tutor_mode: tutorMode,
    conversation_history: conversationHistory,
  };
  if (documentId !== undefined && documentId !== null) {
    body.document_id = documentId;
  }
  if (conversationId !== undefined && conversationId !== null) {
    body.conversation_id = conversationId;
  }
  return fetchJson("/chat", {
    method: "POST",
    headers: {
      ...csrf,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function synthesizeCollaboration({
  message,
  contributions,
  failedAgents = [],
  mode,
  conversationHistory = [],
}) {
  const headers = await csrfHeaders();
  return fetchJson("/collaboration/synthesize", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      contributions,
      failed_agents: failedAgents,
      mode,
      conversation_history: conversationHistory,
    }),
  });
}

export function listStudyDocuments() {
  return fetchJson("/study-documents");
}

export async function getStudyDocumentFile(documentId, { download = false } = {}) {
  let response;
  try {
    const fileUrl =
      `${API_BASE_URL}/study-documents/${encodeURIComponent(documentId)}/file` +
      (download ? "?download=true" : "");
    response = await fetch(
      fileUrl,
      { credentials: "include" },
    );
  } catch {
    throw new Error(
      `Could not connect to QuantaAI at ${API_BASE_URL}. Check that FastAPI is running.`,
    );
  }
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = null;
    }
    const error = new Error(
      typeof detail === "string"
        ? detail
        : `The backend rejected the request (HTTP ${response.status}).`,
    );
    error.status = response.status;
    throw error;
  }
  const contentType = response.headers.get("content-type")
    ?.split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (contentType !== "application/pdf") {
    throw new Error("The document server did not return a PDF file.");
  }

  const file = await response.blob();
  if (file.size === 0) {
    throw new Error("The original PDF is empty or unavailable.");
  }
  const header = new TextDecoder().decode(
    await file.slice(0, 1024).arrayBuffer(),
  );
  if (!header.includes("%PDF-")) {
    throw new Error("The original file is not a valid PDF.");
  }
  return file;
}

export async function uploadStudyDocument(file, onProgress) {
  const csrf = await csrfHeaders();
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/study-documents`);
    request.withCredentials = true;
    request.setRequestHeader("X-CSRF-Token", csrf["X-CSRF-Token"]);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && typeof onProgress === "function") {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });
    request.addEventListener("load", () => {
      let data;
      try {
        data = JSON.parse(request.responseText);
      } catch {
        reject(
          new Error(
            `The backend returned an unreadable response (HTTP ${request.status}).`,
          ),
        );
        return;
      }

      if (request.status < 200 || request.status >= 300) {
        const detail = data?.detail;
        const error = new Error(
          typeof detail === "string"
            ? detail
            : `The backend rejected the request (HTTP ${request.status}).`,
        );
        error.status = request.status;
        reject(error);
        return;
      }
      resolve(data);
    });
    request.addEventListener("error", () => {
      reject(
        new Error(
          `Could not connect to QuantaAI at ${API_BASE_URL}. Check that FastAPI is running.`,
        ),
      );
    });
    request.addEventListener("abort", () => {
      reject(new Error("PDF upload was cancelled."));
    });
    request.send(formData);
  });
}

export async function renameStudyDocument(documentId, filename) {
  const headers = await csrfHeaders();
  return fetchJson(`/study-documents/${encodeURIComponent(documentId)}`, {
    method: "PATCH",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ filename }),
  });
}

export async function deleteStudyDocument(documentId) {
  const csrf = await csrfHeaders();
  return fetchJson(`/study-documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
    headers: csrf,
  });
}

export async function listPlannerTasks() {
  const { tasks } = await fetchJson("/planner/tasks");
  return tasks;
}

export async function createPlannerTask(task) {
  const headers = await csrfHeaders();
  return fetchJson("/planner/tasks", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(task),
  });
}

export async function createPlannerSchedule(scheduleTasks, onTaskCreated) {
  const existingTasks = await listPlannerTasks();
  const identities = new Set(
    existingTasks.map((task) =>
      `${task.title.trim().replace(/\s+/g, " ").toLowerCase()}|${task.due_date}`,
    ),
  );
  const created = [];
  const duplicates = [];
  const failures = [];

  for (const task of scheduleTasks) {
    const identity =
      `${task.title.trim().replace(/\s+/g, " ").toLowerCase()}|${task.due_date}`;
    if (identities.has(identity)) {
      duplicates.push(task);
      continue;
    }

    try {
      const saved = await createPlannerTask(task);
      identities.add(identity);
      created.push(saved);
      onTaskCreated?.(saved);
    } catch (error) {
      if (error.status === 401) throw error;
      failures.push({ task, message: error.message });
    }
  }

  return { created, duplicates, failures };
}

export async function updatePlannerTask(taskId, updates) {
  const headers = await csrfHeaders();
  return fetchJson(`/planner/tasks/${encodeURIComponent(taskId)}`, {
    method: "PATCH",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(updates),
  });
}

export async function deletePlannerTask(taskId) {
  const headers = await csrfHeaders();
  return fetchJson(`/planner/tasks/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
    headers,
  });
}

export async function createQuiz(quiz) {
  const headers = await csrfHeaders();
  return fetchJson("/learning/quizzes", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(quiz),
  });
}

export function listQuizzes() {
  return fetchJson("/learning/quizzes");
}

export function getQuiz(quizId) {
  return fetchJson(`/learning/quizzes/${encodeURIComponent(quizId)}`);
}

export async function submitQuiz(quizId, answers) {
  const headers = await csrfHeaders();
  return fetchJson(`/learning/quizzes/${encodeURIComponent(quizId)}/submit`, {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ answers }),
  });
}

export async function createFlashcardDeck(deck) {
  const headers = await csrfHeaders();
  return fetchJson("/learning/flashcards", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(deck),
  });
}

export function listFlashcardDecks() {
  return fetchJson("/learning/flashcards");
}

export function getFlashcardDeck(deckId) {
  return fetchJson(`/learning/flashcards/${encodeURIComponent(deckId)}`);
}

export async function updateFlashcardProgress(deckId, progress) {
  const headers = await csrfHeaders();
  return fetchJson(
    `/learning/flashcards/${encodeURIComponent(deckId)}/progress`,
    {
      method: "PATCH",
      headers: {
        ...headers,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(progress),
    },
  );
}

export async function deleteFlashcardDeck(deckId) {
  const headers = await csrfHeaders();
  return fetchJson(
    `/learning/flashcards/${encodeURIComponent(deckId)}`,
    {
      method: "DELETE",
      headers,
    },
  );
}

export function getLearningProgress() {
  return fetchJson("/learning/progress");
}

export async function listConversations(limit = 100) {
  const query = new URLSearchParams({ limit: String(limit) });
  return fetchJson(`/conversations?${query.toString()}`);
}

export async function createConversation(title = "New chat") {
  const csrf = await csrfHeaders();
  return fetchJson("/conversations", {
    method: "POST",
    headers: {
      ...csrf,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });
}

export async function getConversation(conversationId, limit = 500) {
  const messages = [];
  let beforeMessageId;
  let conversation;

  do {
    const query = new URLSearchParams({ limit: String(limit) });
    if (beforeMessageId !== undefined) {
      query.set("before_message_id", String(beforeMessageId));
    }

    conversation = await fetchJson(
      `/conversations/${encodeURIComponent(conversationId)}?${query.toString()}`,
    );
    messages.unshift(...conversation.messages);

    if (conversation.has_more) {
      const oldestMessageId = conversation.messages[0]?.id;
      if (!oldestMessageId) {
        throw new Error("The backend returned an incomplete conversation page.");
      }
      beforeMessageId = oldestMessageId;
    } else {
      beforeMessageId = undefined;
    }
  } while (beforeMessageId !== undefined);

  return { ...conversation, messages, has_more: false };
}

export async function appendConversationMessages(conversationId, messages) {
  const csrf = await csrfHeaders();
  return fetchJson(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      headers: {
        ...csrf,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages }),
    },
  );
}

export async function renameConversation(conversationId, title) {
  const csrf = await csrfHeaders();
  return fetchJson(`/conversations/${encodeURIComponent(conversationId)}`, {
    method: "PATCH",
    headers: {
      ...csrf,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(conversationId) {
  const csrf = await csrfHeaders();
  return fetchJson(`/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
    headers: csrf,
  });
}
