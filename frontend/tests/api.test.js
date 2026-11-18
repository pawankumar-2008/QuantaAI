import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import {
  appendConversationMessages,
  createFlashcardDeck,
  createPlannerSchedule,
  createPlannerTask,
  createQuiz,
  createConversation,
  deleteFlashcardDeck,
  deletePlannerTask,
  deleteStudyDocument,
  deleteConversation,
  getLearningProgress,
  getCurrentUser,
  getStudyDocumentFile,
  listFlashcardDecks,
  getConversation,
  listConversations,
  listPlannerTasks,
  listStudyDocuments,
  listQuizzes,
  logoutUser,
  renameConversation,
  renameStudyDocument,
  sendChatMessage,
  synthesizeCollaboration,
  submitQuiz,
  updateCurrentUser,
  updateFlashcardProgress,
  updatePlannerTask,
  uploadStudyDocument,
} from "../src/services/api.js";
import {
  readPreference,
  savePreference,
} from "../src/services/preferences.js";

const originalFetch = globalThis.fetch;
const originalXMLHttpRequest = globalThis.XMLHttpRequest;

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalXMLHttpRequest === undefined) {
    delete globalThis.XMLHttpRequest;
  } else {
    globalThis.XMLHttpRequest = originalXMLHttpRequest;
  }
});

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("lists conversations with cookie credentials", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    return jsonResponse({ conversations: [], has_more: false });
  };

  const result = await listConversations(25);

  assert.deepEqual(result.conversations, []);
  assert.match(calls[0].url, /\/conversations\?limit=25$/);
  assert.equal(calls[0].options.credentials, "include");
});

test("loads every message page in chronological order", async () => {
  const pages = [
    { messages: [{ id: 5 }, { id: 6 }], has_more: true },
    { messages: [{ id: 3 }, { id: 4 }], has_more: true },
    { messages: [{ id: 1 }, { id: 2 }], has_more: false },
  ];
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(String(url));
    const page = pages.shift();
    return jsonResponse({ id: 12, title: "Test", ...page });
  };

  const result = await getConversation(12, 2);

  assert.deepEqual(result.messages.map(({ id }) => id), [1, 2, 3, 4, 5, 6]);
  assert.equal(result.has_more, false);
  assert.equal(calls.length, 3);
  assert.match(calls[1], /before_message_id=5/);
  assert.match(calls[2], /before_message_id=3/);
});

test("sends CSRF-protected requests for message saves, rename and delete", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "test-csrf" });
    if (path.endsWith("/messages")) return jsonResponse({ messages: [] });
    if (options.method === "DELETE") return new Response(null, { status: 204 });
    return jsonResponse({ id: 12, title: "Updated" });
  };

  await appendConversationMessages(12, [
    { role: "user", content: "Hello", agent: "Study" },
    {
      role: "assistant",
      content: "An experimental answer.",
      agent: "Study",
      model: "QuantaLM",
      provider: "QuantaLM Offline",
      experimental: true,
      reliability_warning: true,
      document_source: "uploaded_pdf",
      document_pages: [1, 3],
    },
  ]);
  await createConversation("New chat");
  await renameConversation(12, "Updated");
  await deleteConversation(12);

  const writes = calls.filter(({ path }) => path !== "/auth/csrf");
  assert.deepEqual(
    writes.map(({ options }) => options.method),
    ["POST", "POST", "PATCH", "DELETE"],
  );
  for (const { options } of writes) {
    assert.equal(options.credentials, "include");
    assert.equal(options.headers["X-CSRF-Token"], "test-csrf");
  }
  assert.equal(
    JSON.parse(writes[1].options.body).title,
    "New chat",
  );
  assert.deepEqual(
    JSON.parse(writes[0].options.body).messages[1],
    {
      role: "assistant",
      content: "An experimental answer.",
      agent: "Study",
      model: "QuantaLM",
      provider: "QuantaLM Offline",
      experimental: true,
      reliability_warning: true,
      document_source: "uploaded_pdf",
      document_pages: [1, 3],
    },
  );
  assert.equal(
    JSON.parse(writes[2].options.body).title,
    "Updated",
  );
});

test("reports the backend status and detail for failed API requests", async () => {
  globalThis.fetch = async () =>
    jsonResponse({ detail: "Not found" }, 404);

  await assert.rejects(listConversations(), (error) => {
    assert.equal(error.status, 404);
    assert.equal(error.message, "Not found");
    return true;
  });
});

test("sends recent conversation context with a chat request", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "chat-csrf" });
    return jsonResponse({
      reply: "Follow-up response",
      videos: [],
      document_source: "uploaded_pdf",
      document_pages: [2],
    });
  };
  const conversationHistory = [
    { role: "user", content: "Explain normalization." },
    { role: "assistant", content: "It reduces redundant data." },
  ];

  const result = await sendChatMessage({
    message: "Show me an example.",
    agent: "Study",
    mode: "teacher",
    tutorMode: "socratic",
    conversationHistory,
    documentId: 23,
  });

  const request = calls.find(({ path }) => path === "/chat");
  assert.equal(request.options.headers["X-CSRF-Token"], "chat-csrf");
  assert.equal(result.document_source, "uploaded_pdf");
  assert.deepEqual(result.document_pages, [2]);
  assert.deepEqual(JSON.parse(request.options.body), {
    message: "Show me an example.",
    agent: "Study",
    mode: "teacher",
    tutor_mode: "socratic",
    conversation_history: conversationHistory,
    document_id: 23,
  });
});

test("sends Surface conversation identity for conversation-scoped PDF context", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "surface-csrf" });
    return jsonResponse({ reply: "Document answer" });
  };

  await sendChatMessage({
    message: "Summarize this document.",
    agent: "Surface",
    mode: "teacher",
    conversationId: 42,
  });

  const request = calls.find(({ path }) => path === "/chat");
  assert.deepEqual(JSON.parse(request.options.body), {
    message: "Summarize this document.",
    agent: "Surface",
    mode: "teacher",
    tutor_mode: "direct",
    conversation_history: [],
    conversation_id: 42,
  });
});

test("synthesizes collaboration contributions with CSRF and conversation context", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "collab-csrf" });
    return jsonResponse({
      reply: "A tree is both hierarchical and acyclic.",
      provider: "Google Gemini",
      model: "gemini-test-model",
    });
  };
  const conversationHistory = [
    { role: "user", content: "Explain trees." },
    { role: "assistant", content: "A tree is hierarchical." },
  ];

  const result = await synthesizeCollaboration({
    message: "Compare the explanations.",
    contributions: [
      { agent: "Study", response: "A tree is hierarchical." },
      { agent: "Coding", response: "A tree is acyclic." },
    ],
    failedAgents: ["Career"],
    mode: "teacher",
    conversationHistory,
  });

  const request = calls.find(({ path }) => path === "/collaboration/synthesize");
  assert.equal(result.provider, "Google Gemini");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.credentials, "include");
  assert.equal(request.options.headers["X-CSRF-Token"], "collab-csrf");
  assert.deepEqual(JSON.parse(request.options.body), {
    message: "Compare the explanations.",
    contributions: [
      { agent: "Study", response: "A tree is hierarchical." },
      { agent: "Coding", response: "A tree is acyclic." },
    ],
    failed_agents: ["Career"],
    mode: "teacher",
    conversation_history: conversationHistory,
  });
});

test("lists and removes study documents with authenticated requests", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "doc-csrf" });
    if (options.method === "DELETE") return new Response(null, { status: 204 });
    return jsonResponse([{ id: 8, filename: "notes.pdf", status: "indexed" }]);
  };

  const documents = await listStudyDocuments();
  await deleteStudyDocument(8);

  assert.equal(documents[0].filename, "notes.pdf");
  assert.deepEqual(
    calls.map(({ path }) => path),
    ["/study-documents", "/auth/csrf", "/study-documents/8"],
  );
  assert.equal(calls[0].options.credentials, "include");
  assert.equal(calls[2].options.method, "DELETE");
  assert.equal(calls[2].options.headers["X-CSRF-Token"], "doc-csrf");
});

test("renames and previews uploaded documents through authenticated APIs", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    const requestUrl = new URL(String(url), "http://localhost");
    calls.push({ path, search: requestUrl.search, options });
    if (path === "/auth/csrf") {
      return jsonResponse({ csrf_token: "documents-csrf" });
    }
    if (path.endsWith("/file")) {
      return new Response("%PDF-1.4\nvalid pdf content", {
        status: 200,
        headers: { "Content-Type": "application/pdf" },
      });
    }
    return jsonResponse({ id: 8, filename: "renamed.pdf" });
  };

  await renameStudyDocument(8, "renamed.pdf");
  const file = await getStudyDocumentFile(8);

  assert.equal(await file.text(), "%PDF-1.4\nvalid pdf content");
  const downloadableFile = await getStudyDocumentFile(8, { download: true });
  assert.equal(await downloadableFile.text(), "%PDF-1.4\nvalid pdf content");
  const rename = calls.find(({ options }) => options.method === "PATCH");
  assert.equal(rename.path, "/study-documents/8");
  assert.equal(rename.options.headers["X-CSRF-Token"], "documents-csrf");
  assert.deepEqual(JSON.parse(rename.options.body), { filename: "renamed.pdf" });
  const fileRequests = calls.filter(({ path }) => path.endsWith("/file"));
  assert.equal(fileRequests[0].options.credentials, "include");
  assert.equal(fileRequests[0].search, "");
  assert.equal(fileRequests[1].search, "?download=true");
});

test("rejects unavailable or invalid PDF file responses", async () => {
  globalThis.fetch = async () =>
    jsonResponse({ detail: "The original PDF is not available." }, 404);
  await assert.rejects(getStudyDocumentFile(8), {
    status: 404,
    message: "The original PDF is not available.",
  });

  globalThis.fetch = async () =>
    new Response("not a pdf", {
      status: 200,
      headers: { "Content-Type": "application/pdf" },
    });
  await assert.rejects(getStudyDocumentFile(8), {
    message: "The original file is not a valid PDF.",
  });

  globalThis.fetch = async () =>
    new Response("%PDF-1.4\ncontent", {
      status: 200,
      headers: { "Content-Type": "text/plain" },
    });
  await assert.rejects(getStudyDocumentFile(8), {
    message: "The document server did not return a PDF file.",
  });
});

test("uploads a PDF with cookie credentials, CSRF, and progress reporting", async () => {
  const progressHandlers = [];
  const requestCalls = [];
  class MockXMLHttpRequest {
    constructor() {
      this.upload = {
        addEventListener: (name, handler) => {
          if (name === "progress") progressHandlers.push(handler);
        },
      };
      this.listeners = {};
      requestCalls.push(this);
    }

    open(method, url) {
      this.method = method;
      this.url = url;
    }

    setRequestHeader(name, value) {
      this.headers = { ...this.headers, [name]: value };
    }

    addEventListener(name, handler) {
      this.listeners[name] = handler;
    }

    send(body) {
      this.body = body;
      progressHandlers[0]({
        lengthComputable: true,
        loaded: 25,
        total: 100,
      });
      this.status = 201;
      this.responseText = JSON.stringify({
        id: 8,
        filename: "notes.pdf",
        status: "indexed",
      });
      this.listeners.load();
    }
  }
  globalThis.XMLHttpRequest = MockXMLHttpRequest;
  globalThis.fetch = async () =>
    jsonResponse({ csrf_token: "upload-csrf" });
  const progress = [];

  const uploaded = await uploadStudyDocument(
    new Blob(["%PDF-1.4"], { type: "application/pdf" }),
    (value) => progress.push(value),
  );

  assert.equal(uploaded.status, "indexed");
  assert.deepEqual(progress, [25]);
  assert.equal(requestCalls[0].method, "POST");
  assert.match(requestCalls[0].url, /\/study-documents$/);
  assert.equal(requestCalls[0].withCredentials, true);
  assert.equal(requestCalls[0].headers["X-CSRF-Token"], "upload-csrf");
  assert.equal(requestCalls[0].body.get("file").type, "application/pdf");
});

test("loads the authenticated profile from the existing user endpoint", async () => {
  const user = {
    id: 42,
    name: "Ada Lovelace",
    email: "ada@example.com",
    profile: {},
  };
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url: String(url), options };
    return jsonResponse(user);
  };

  assert.deepEqual(await getCurrentUser(), user);
  assert.match(request.url, /\/auth\/me$/);
  assert.equal(request.options.credentials, "include");
});

test("updates the profile and persists planner task CRUD through CSRF-protected requests", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "account-csrf" });
    if (path === "/auth/me") {
      return jsonResponse({
        id: 42,
        name: "Ada Lovelace",
        profile: { learningGoal: "Algorithms" },
      });
    }
    if (options.method === "DELETE") return new Response(null, { status: 204 });
    if (path === "/planner/tasks") {
      return jsonResponse({ tasks: [{ id: 7, title: "Review" }] });
    }
    return jsonResponse({
      id: 7,
      title: "Practice",
      due_date: "2026-03-10",
      completed: true,
    });
  };

  await updateCurrentUser({
    name: "Ada Lovelace",
    profile: { learningGoal: "Algorithms" },
  });
  assert.deepEqual(await listPlannerTasks(), [{ id: 7, title: "Review" }]);
  await createPlannerTask({
    title: "Review",
    description: "Review the key concepts.",
    due_date: "2026-03-10",
  });
  await updatePlannerTask(7, { completed: true });
  await deletePlannerTask(7);

  const writes = calls.filter(({ path }) => path !== "/auth/csrf");
  assert.deepEqual(
    writes.map(({ path, options }) => [path, options.method]),
    [
      ["/auth/me", "PATCH"],
      ["/planner/tasks", undefined],
      ["/planner/tasks", "POST"],
      ["/planner/tasks/7", "PATCH"],
      ["/planner/tasks/7", "DELETE"],
    ],
  );
  assert.equal(writes[0].options.headers["X-CSRF-Token"], "account-csrf");
  assert.deepEqual(JSON.parse(writes[2].options.body), {
    title: "Review",
    description: "Review the key concepts.",
    due_date: "2026-03-10",
  });
  assert.equal(writes[2].options.headers["X-CSRF-Token"], "account-csrf");
  assert.deepEqual(
    JSON.parse(writes[0].options.body),
    { name: "Ada Lovelace", profile: { learningGoal: "Algorithms" } },
  );
  assert.equal(writes[3].options.headers["X-CSRF-Token"], "account-csrf");
});

test("saves generated schedule tasks through the owned task API and skips duplicates", async () => {
  const calls = [];
  const taskCreatedEvents = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "schedule-csrf" });
    if (path === "/planner/tasks" && !options.method) {
      return jsonResponse({
        tasks: [{
          id: 5,
          title: "Existing revision",
          due_date: "2026-11-14",
        }],
      });
    }
    if (path === "/planner/tasks" && options.method === "POST") {
      const task = JSON.parse(options.body);
      if (task.title === "Failing task") {
        return jsonResponse({ detail: "Planner storage unavailable" }, 503);
      }
      return jsonResponse({ id: 6, ...task, completed: false }, 201);
    }
    throw new Error(`Unexpected request: ${path}`);
  };

  const result = await createPlannerSchedule(
    [
      {
        title: " Existing revision ",
        description: "Already in the planner.",
        due_date: "2026-11-14",
      },
      {
        title: "Daily practice",
        description: "Solve today's practice set.",
        due_date: "2026-11-13",
      },
      {
        title: "daily   practice",
        description: "Duplicate within the same request.",
        due_date: "2026-11-13",
      },
      {
        title: "Weekend revision",
        description: "Revise the week's concepts on Saturday.",
        due_date: "2026-11-14",
      },
      {
        title: "Weekend mini-test",
        description: "Check retention on Sunday.",
        due_date: "2026-11-15",
      },
      {
        title: "Failing task",
        description: "Storage failure should be reported.",
        due_date: "2026-11-15",
      },
    ],
    (task) => taskCreatedEvents.push(task),
  );

  assert.equal(result.created.length, 3);
  assert.equal(result.duplicates.length, 2);
  assert.equal(result.failures.length, 1);
  assert.equal(result.failures[0].message, "Planner storage unavailable");
  assert.equal(taskCreatedEvents.length, 3);
  assert.deepEqual(
    calls
      .filter(({ options }) => options.method === "POST")
      .map(({ options }) => {
        const { title, due_date } = JSON.parse(options.body);
        return [title, due_date];
      }),
    [
      ["Daily practice", "2026-11-13"],
      ["Weekend revision", "2026-11-14"],
      ["Weekend mini-test", "2026-11-15"],
      ["Failing task", "2026-11-15"],
    ],
  );
  assert.ok(
    calls
      .filter(({ options }) => options.method === "POST")
      .every(({ options }) => options.headers["X-CSRF-Token"] === "schedule-csrf"),
  );
});

test("does not create schedule tasks when the existing-task check fails", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    return jsonResponse({ detail: "Planner list unavailable" }, 503);
  };

  await assert.rejects(
    createPlannerSchedule([
      { title: "Daily practice", description: "Practice.", due_date: "2026-11-14" },
    ]),
    /Planner list unavailable/,
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/planner/tasks");
});

test("retries an already saved schedule without duplicating tasks or shifting dates", async () => {
  const savedTasks = [];
  const createdInUi = [];
  let nextId = 50;
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "retry-csrf" });
    if (path === "/planner/tasks" && !options.method) {
      return jsonResponse({ tasks: [...savedTasks] });
    }
    if (path === "/planner/tasks" && options.method === "POST") {
      const task = { id: nextId++, ...JSON.parse(options.body), completed: false };
      savedTasks.push(task);
      return jsonResponse(task, 201);
    }
    throw new Error(`Unexpected request: ${path}`);
  };
  const generatedTasks = [
    {
      title: "Study normalization",
      description: "Review DBMS normalization.",
      due_date: "2026-11-13",
    },
    {
      title: "Solve DBMS PYQs",
      description: "Solve twenty questions.",
      due_date: "2026-11-14",
    },
  ];

  const first = await createPlannerSchedule(
    generatedTasks,
    (task) => createdInUi.push(task),
  );
  const retry = await createPlannerSchedule(
    generatedTasks,
    (task) => createdInUi.push(task),
  );

  assert.deepEqual(first.created.map(({ due_date }) => due_date), [
    "2026-11-13",
    "2026-11-14",
  ]);
  assert.equal(first.created.length, 2);
  assert.equal(retry.created.length, 0);
  assert.equal(retry.duplicates.length, 2);
  assert.equal(savedTasks.length, 2);
  assert.equal(createdInUi.length, 2);
});

test("sends quiz and flashcard operations with user responses", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "learning-csrf" });
    if (options.method === "DELETE") return new Response(null, { status: 204 });
    if (path === "/learning/quizzes") {
      return jsonResponse({ quizzes: [], id: 12, questions: [] });
    }
    if (path.endsWith("/submit")) {
      return jsonResponse({ id: 12, score: 2 });
    }
    if (path === "/learning/progress") {
      return jsonResponse({ completed_quizzes: 0 });
    }
    if (path.endsWith("/progress")) {
      return jsonResponse({ id: 9, current_index: 1 });
    }
    if (path === "/learning/flashcards") {
      return jsonResponse({ decks: [], id: 9, cards: [] });
    }
    return jsonResponse({});
  };

  await createQuiz({
    topic: "Recursion",
    question_count: 3,
    difficulty: "medium",
    mode: "teacher",
  });
  await createQuiz({
    document_id: 23,
    question_count: 5,
    difficulty: "basic",
    mode: "teacher",
  });
  await submitQuiz(12, [0, 1, 0]);
  assert.deepEqual((await listQuizzes()).quizzes, []);
  await createFlashcardDeck({
    topic: "Trees",
    card_count: 3,
    difficulty: "medium",
    mode: "offline",
  });
  await createFlashcardDeck({
    document_id: 23,
    card_count: 3,
    difficulty: "basic",
    mode: "teacher",
  });
  assert.deepEqual((await listFlashcardDecks()).decks, []);
  await updateFlashcardProgress(9, { current_index: 1, known: true });
  await deleteFlashcardDeck(9);
  assert.equal((await getLearningProgress()).completed_quizzes, 0);

  const quizRequest = calls.find(({ path }) => path === "/learning/quizzes" && path);
  const quizSubmission = calls.find(({ path }) => path.endsWith("/submit"));
  const flashcardRequests = calls.filter(
    ({ path, options }) => path === "/learning/flashcards" && options.method === "POST",
  );
  assert.deepEqual(JSON.parse(quizRequest.options.body), {
    topic: "Recursion",
    question_count: 3,
    difficulty: "medium",
    mode: "teacher",
  });
  const documentQuizRequest = calls.find(
    ({ path, options }) =>
      path === "/learning/quizzes" &&
      JSON.parse(options.body).document_id === 23,
  );
  assert.deepEqual(JSON.parse(documentQuizRequest.options.body), {
    document_id: 23,
    question_count: 5,
    difficulty: "basic",
    mode: "teacher",
  });
  assert.deepEqual(JSON.parse(quizSubmission.options.body), { answers: [0, 1, 0] });
  assert.deepEqual(JSON.parse(flashcardRequests[0].options.body), {
    topic: "Trees",
    card_count: 3,
    difficulty: "medium",
    mode: "offline",
  });
  assert.deepEqual(JSON.parse(flashcardRequests[1].options.body), {
    document_id: 23,
    card_count: 3,
    difficulty: "basic",
    mode: "teacher",
  });
  assert.ok(calls.every(({ options }) => options.credentials === "include"));
});

test("invalidates the authenticated session with a CSRF-protected logout", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), "http://localhost").pathname;
    calls.push({ path, options });
    if (path === "/auth/csrf") return jsonResponse({ csrf_token: "logout-csrf" });
    return new Response(null, { status: 204 });
  };

  await logoutUser();

  assert.deepEqual(calls.map(({ path }) => path), ["/auth/csrf", "/auth/logout"]);
  const logoutRequest = calls[1].options;
  assert.equal(logoutRequest.method, "POST");
  assert.equal(logoutRequest.credentials, "include");
  assert.equal(logoutRequest.headers["X-CSRF-Token"], "logout-csrf");
});

test("persists only supported theme and interaction preferences", () => {
  const values = new Map();
  const originalWindow = globalThis.window;
  globalThis.window = {
    localStorage: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
    },
  };

  try {
    savePreference("quantaaI.theme", "dark");
    savePreference("quantaaI.defaultInteractionMode", "offline");

    assert.equal(readPreference("quantaaI.theme", ["light", "dark"], "light"), "dark");
    assert.equal(
      readPreference(
        "quantaaI.defaultInteractionMode",
        ["auto", "teacher", "offline"],
        "auto",
      ),
      "offline",
    );
    savePreference("quantaaI.theme", "unsupported");
    assert.equal(readPreference("quantaaI.theme", ["light", "dark"], "light"), "light");
  } finally {
    if (originalWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = originalWindow;
    }
  }
});
