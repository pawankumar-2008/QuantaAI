
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  BriefcaseBusiness,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Clock3,
  Code2,
  Compass,
  FileText,
  FolderOpen,
  ListChecks,
  Paperclip,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Settings as SettingsIcon,
  Sparkles,
  Trash2,
  UsersRound,
} from "lucide-react";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";
import "./App.css";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Settings from "./pages/Settings";
import Profile from "./pages/Profile";
import Planner from "./pages/Planner";
import Quiz from "./pages/Quiz";
import Flashcards from "./pages/Flashcards";
import Progress from "./pages/Progress";
import DocumentLibrary from "./pages/DocumentLibrary";
import ProcessingIndicator from "./components/ProcessingIndicator";
import MarkdownReply from "./components/MarkdownReply";
import { readPreference, savePreference } from "./services/preferences";
import {
  getUpcomingPlannerTasks,
  loadPlannerReminderTimes,
  savePlannerReminderTime,
  takeDuePlannerReminders,
} from "./services/plannerReminders";
import frontendPackage from "../package.json";
import {
  getBackendHealth,
  appendConversationMessages,
  createPlannerSchedule,
  createPlannerTask,
  createConversation,
  getCurrentUser,
  getLearningProgress,
  getConversation,
  listConversations,
  listPlannerTasks,
  loginUser,
  logoutUser,
  deleteConversation as deleteConversationRequest,
  deletePlannerTask,
  deleteStudyDocument,
  renameConversation as renameConversationRequest,
  registerUser,
  listStudyDocuments,
  sendChatMessage,
  synthesizeCollaboration,
  updateCurrentUser,
  updatePlannerTask,
  uploadStudyDocument,
} from "./services/api";

const agents = [
  {
    id: "Study",
    icon: BookOpen,
    name: "Study Agent",
    desc: "Learn concepts clearly",
  },
  {
    id: "Coding",
    icon: Code2,
    name: "Coding Agent",
    desc: "Practice and debug code",
  },
  {
    id: "Career",
    icon: BriefcaseBusiness,
    name: "Career Agent",
    desc: "Prepare for your career",
  },
  {
    id: "Guide",
    icon: Compass,
    name: "Guide Agent",
    desc: "Explore new topics",
  },
  {
    id: "Planner",
    icon: CalendarDays,
    name: "Planner Agent",
    desc: "Organize your learning",
  },
  {
    id: "Exam",
    icon: ClipboardCheck,
    name: "Exam Agent",
    desc: "Prepare for examinations",
  },
  {
    id: "Research",
    icon: Search,
    name: "Research Agent",
    desc: "Explore ideas and evidence",
  },
];

const surfaceAssistant = {
  id: "Surface",
  icon: Sparkles,
  name: "Surface AI",
  desc: "A general-purpose assistant for learning, planning, and ideas",
};

const modes = [
  { value: "auto", label: "AUTO" },
  { value: "teacher", label: "ONLINE" },
  { value: "offline", label: "OFFLINE" },
];

const pageTitles = {
  settings: "Settings",
  profile: "My Profile",
  planner: "Planner & To-do",
  quiz: "Quiz",
  flashcards: "Flashcards",
  progress: "Learning progress",
  documents: "My Uploads",
};

const suggestions = [
  {
    icon: BookOpen,
    title: "Explain a concept",
    text: "Teach me a difficult topic step by step.",
    prompt: "Explain a difficult technical concept step by step.",
  },
  {
    icon: CalendarDays,
    title: "Plan my study week",
    text: "Create a realistic learning schedule.",
    prompt: "Help me plan my study week.",
  },
  {
    icon: Code2,
    title: "Practice coding",
    text: "Help me solve a programming problem.",
    prompt:
      "Help me practice programming with a beginner-friendly problem.",
  },
  {
    icon: Compass,
    title: "Explore careers",
    text: "Find skills for my career goals.",
    prompt: "Help me explore technology career paths.",
  },
];

function formatMessageTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function shouldSearchOnline(message, mode) {
  return (
    mode !== "offline" &&
    /\b(search (?:the )?(?:web|internet)|look (?:this )?up|web research|research (?:this|the|current)|find (?:current )?sources|give me sources|current|currently|latest|recent|today|now|this year|202[5-9]|price|release|version|availability|schedule|deadline|documentation|docs|api reference|official (?:guide|docs)|library version|framework version|verify|fact.check|fact check|confirm|validate|source this|cite sources|with citations)\b/i.test(
      message,
    )
  );
}

function formatConversationMessage(message) {
  const agent = agents.find((item) => item.id === message.agent);
  const isCollaboration = message.agent === "Collaboration";
  return {
    id: String(message.id),
    role: message.role,
    text: message.content,
    agent: isCollaboration
      ? "Collaboration Summary"
      : message.agent === surfaceAssistant.id
        ? surfaceAssistant.name
        : agent?.name || message.agent || "Surface AI",
    icon: isCollaboration
      ? UsersRound
      : message.agent === surfaceAssistant.id
        ? surfaceAssistant.icon
        : agent?.icon,
    model: message.model || "",
    provider: message.provider || "",
    experimental: message.experimental === true,
    reliabilityWarning: message.reliability_warning === true,
    timestamp: message.created_at,
    videos: Array.isArray(message.videos) ? message.videos : [],
    videoMessage: message.video_message || message.videoMessage || "",
    searchPerformed: message.search_performed === true,
    searchAgent: message.search_agent || "",
    searchMessage: message.search_message || "",
    sources: Array.isArray(message.sources) ? message.sources : [],
    documentSource: message.document_source || "",
    documentPages: Array.isArray(message.document_pages)
      ? message.document_pages
      : [],
    documentAttachments: Array.isArray(message.document_attachments)
      ? message.document_attachments
      : [],
  };
}

function titleFromMessage(message) {
  const title = message.replace(/\s+/g, " ").trim();
  return title.length > 72 ? `${title.slice(0, 69)}…` : title;
}

function sortPlannerTasks(items) {
  return [...items].sort((left, right) => {
    if (left.completed !== right.completed) {
      return left.completed ? 1 : -1;
    }
    return left.due_date.localeCompare(right.due_date) || left.id - right.id;
  });
}

function formatPlannerDate(value) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(
    new Date(`${value}T00:00:00`),
  );
}

function App() {
  // --------------------------------------------------
  // 1. ALL STATE HOOKS MUST BE DECLARED UNCONDITIONALLY
  // --------------------------------------------------

  const [currentScreen, setCurrentScreen] = useState("login");
  const [authStatus, setAuthStatus] = useState("checking");
  const [user, setUser] = useState(null);

  const [activeAgent, setActiveAgent] = useState("Surface");
  const [selectedAgents, setSelectedAgents] = useState([]);
  const [collaborationMode, setCollaborationMode] = useState(false);
  const [respondingAgents, setRespondingAgents] = useState([]);
  const [searchingAgents, setSearchingAgents] = useState([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [chatMode, setChatMode] = useState(() =>
    readPreference(
      "quantaaI.defaultInteractionMode",
      modes.map((mode) => mode.value),
      "auto",
    ),
  );
  const [tutorMode, setTutorMode] = useState(() =>
    readPreference(
      "quantaaI.tutorMode",
      ["direct", "socratic", "deep_learning"],
      "direct",
    ),
  );
  const [theme, setTheme] = useState(() =>
    readPreference("quantaaI.theme", ["light", "dark"], "light"),
  );
  const [activePage, setActivePage] = useState("workspace");
  const [studyDocumentDeleteConfirmationId, setStudyDocumentDeleteConfirmationId] = useState(null);
  const [quizLaunchRequest, setQuizLaunchRequest] = useState(null);
  const [flashcardDeckToReview, setFlashcardDeckToReview] = useState(null);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [accountMenuError, setAccountMenuError] = useState("");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [recentChatsExpanded, setRecentChatsExpanded] = useState(false);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [studyDocuments, setStudyDocuments] = useState([]);
  const [selectedStudyDocumentId, setSelectedStudyDocumentId] = useState(null);
  const [studyDocumentMenuOpen, setStudyDocumentMenuOpen] = useState(false);
  const [studyDocumentPickerOpen, setStudyDocumentPickerOpen] = useState(false);
  const [studyDocumentsLoading, setStudyDocumentsLoading] = useState(false);
  const [studyDocumentUploading, setStudyDocumentUploading] = useState(false);
  const [studyDocumentProgress, setStudyDocumentProgress] = useState(0);
  const [studyDocumentError, setStudyDocumentError] = useState("");
  const [studyDocumentMutationId, setStudyDocumentMutationId] = useState(null);
  const [surfaceAttachments, setSurfaceAttachments] = useState([]);
  const [surfaceDocumentUploading, setSurfaceDocumentUploading] = useState(false);
  const [surfaceDocumentProgress, setSurfaceDocumentProgress] = useState(0);
  const [surfaceUploadingFilename, setSurfaceUploadingFilename] = useState("");
  const [surfaceDocumentError, setSurfaceDocumentError] = useState("");
  const [conversationMutation, setConversationMutation] = useState("");
  const [renamingConversationId, setRenamingConversationId] = useState(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState(null);
  const selectionRequestId = useRef(0);
  const historyRequestId = useRef(0);
  const accountMenuRef = useRef(null);
  const studyDocumentInputRef = useRef(null);

  const [tasks, setTasks] = useState([]);
  const [taskTitle, setTaskTitle] = useState("");
  const [taskDate, setTaskDate] = useState("");
  const [showTasks, setShowTasks] = useState(false);
  const [plannerLoading, setPlannerLoading] = useState(false);
  const [plannerError, setPlannerError] = useState("");
  const [taskMutation, setTaskMutation] = useState("");
  const [plannerReminderTimes, setPlannerReminderTimes] = useState({});
  const [plannerReminderError, setPlannerReminderError] = useState("");
  const [reminderAlerts, setReminderAlerts] = useState([]);
  const [browserNotificationsEnabled, setBrowserNotificationsEnabled] = useState(
    () => typeof Notification !== "undefined" && Notification.permission === "granted",
  );
  const [learningProgress, setLearningProgress] = useState(null);
  const [progressError, setProgressError] = useState("");

  const [notice, setNotice] = useState(
    "Your personal learning workspace"
  );

  const [busy, setBusy] = useState(false);

  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [backendMessage, setBackendMessage] = useState("");
  const [teacherAvailable, setTeacherAvailable] = useState(null);
  const [youtubeAvailable, setYoutubeAvailable] = useState(false);
  const [quantaLoaded, setQuantaLoaded] = useState(false);

  const selectedAgent =
    activeAgent === surfaceAssistant.id
      ? surfaceAssistant
      : agents.find((agent) => agent.id === activeAgent) || surfaceAssistant;
  const selectedStudyDocument = studyDocuments.find(
    (document) => document.id === selectedStudyDocumentId,
  );
  const upcomingTasks = getUpcomingPlannerTasks(tasks);
  const activeParticipants = collaborationMode
    ? selectedAgents
        .map((agentId) => agents.find((agent) => agent.id === agentId))
        .filter(Boolean)
    : activeAgent === surfaceAssistant.id
      ? [surfaceAssistant]
      : [selectedAgent];
  const recentChatsVisible = !sidebarCollapsed && recentChatsExpanded;

  useEffect(() => {
    savePreference("quantaaI.theme", theme);
  }, [theme]);

  useEffect(() => {
    savePreference("quantaaI.defaultInteractionMode", chatMode);
  }, [chatMode]);

  useEffect(() => {
    savePreference("quantaaI.tutorMode", tutorMode);
  }, [tutorMode]);

  useEffect(() => {
    if (!accountMenuOpen) return undefined;

    function closeAccountMenu(event) {
      if (!accountMenuRef.current?.contains(event.target)) {
        setAccountMenuOpen(false);
        setAccountMenuError("");
      }
    }

    function handleAccountMenuKeyDown(event) {
      if (event.key === "Escape") {
        setAccountMenuOpen(false);
        setAccountMenuError("");
      }
    }

    window.addEventListener("pointerdown", closeAccountMenu);
    window.addEventListener("keydown", handleAccountMenuKeyDown);
    return () => {
      window.removeEventListener("pointerdown", closeAccountMenu);
      window.removeEventListener("keydown", handleAccountMenuKeyDown);
    };
  }, [accountMenuOpen]);

  const handleAuthenticationExpired = useCallback(() => {
    selectionRequestId.current += 1;
    historyRequestId.current += 1;
    setUser(null);
    setReminderAlerts([]);
    setAuthStatus("unauthenticated");
    setCurrentScreen("login");
    setAccountMenuOpen(false);
    setAccountMenuError("");
    setMessages([]);
    setTasks([]);
    setPlannerReminderTimes({});
    setPlannerReminderError("");
    setPlannerLoading(false);
    setPlannerError("");
    setTaskMutation("");
    setLearningProgress(null);
    setProgressError("");
    setStudyDocuments([]);
    setSelectedStudyDocumentId(null);
    setStudyDocumentsLoading(false);
    setStudyDocumentError("");
    setStudyDocumentUploading(false);
    setStudyDocumentProgress(0);
    setStudyDocumentMutationId(null);
    setStudyDocumentDeleteConfirmationId(null);
    setSurfaceAttachments([]);
    setSurfaceDocumentError("");
    setConversations([]);
    setActiveConversationId(null);
    setConversationLoading(false);
    setConversationMutation("");
    setBusy(false);
    setRespondingAgents([]);
    setHistoryLoading(false);
    setHistoryError("");
  }, []);

  useEffect(() => {
    if (
      authStatus !== "authenticated" ||
      !["Study", surfaceAssistant.id].includes(activeAgent) ||
      collaborationMode
    ) {
      return undefined;
    }
    let cancelled = false;
    listStudyDocuments()
      .then((documents) => {
        if (cancelled) return;
        setStudyDocumentError("");
        setStudyDocuments(documents);
        if (activeAgent === "Study") {
          setSelectedStudyDocumentId((current) =>
            documents.some(
              (document) =>
                document.id === current && document.status === "indexed",
            )
              ? current
              : null,
          );
        }
      })
      .catch((error) => {
        if (cancelled) return;
        if (error.status === 401) {
          handleAuthenticationExpired();
        } else {
          setStudyDocumentError(
            error.message || "Unable to load your study documents.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setStudyDocumentsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    activeAgent,
    authStatus,
    collaborationMode,
    handleAuthenticationExpired,
    user?.id,
  ]);

  const refreshConversationList = useCallback(
    async (isCancelled = () => false) => {
      const requestId = historyRequestId.current + 1;
      historyRequestId.current = requestId;
      try {
        const { conversations: loadedConversations } =
          await listConversations();
        if (!isCancelled() && historyRequestId.current === requestId) {
          setConversations(loadedConversations);
        }
      } catch (error) {
        if (isCancelled() || historyRequestId.current !== requestId) return;
        if (error.status === 401) {
          handleAuthenticationExpired();
        } else {
          setHistoryError(
            error.message || "Unable to load your conversations.",
          );
        }
      } finally {
        if (!isCancelled() && historyRequestId.current === requestId) {
          setHistoryLoading(false);
        }
      }
    },
    [handleAuthenticationExpired],
  );

  const refreshLearningProgress = useCallback(async () => {
    try {
      const progress = await getLearningProgress();
      setLearningProgress(progress);
      setProgressError("");
      return true;
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setProgressError(
          error.message || "Unable to refresh learning progress.",
        );
      }
      return false;
    }
  }, [handleAuthenticationExpired]);

  const handleDocumentsChanged = useCallback((documents) => {
    const availableDocuments = new Map(
      documents.map((document) => [document.id, document]),
    );
    setStudyDocuments(documents);
    setSelectedStudyDocumentId((current) =>
      documents.some(
        (document) => document.id === current && document.status === "indexed",
      )
        ? current
        : null,
    );
    setSurfaceAttachments((current) =>
      current.flatMap((attachment) => {
        const document = availableDocuments.get(attachment.id);
        return document?.status === "indexed"
          ? [{ ...attachment, filename: document.filename }]
          : [];
      }),
    );
    setMessages((current) =>
      current.map((message) => ({
        ...message,
        documentAttachments: message.documentAttachments.map((attachment) => {
          const document = availableDocuments.get(attachment.id);
          return document
            ? { ...attachment, filename: document.filename }
            : attachment;
        }),
      })),
    );
  }, []);

  const launchQuizFromDeck = useCallback((deck) => {
    setQuizLaunchRequest({
      topic: deck.topic,
      difficulty: deck.difficulty || "basic",
      source_type: deck.source_type,
      document_id: deck.document_id,
      deck_id: deck.id,
    });
    setActivePage("quiz");
  }, []);

  const clearQuizLaunchRequest = useCallback(() => {
    setQuizLaunchRequest(null);
  }, []);

  // --------------------------------------------------
  // 2. BACKEND HEALTH CHECK
  // This hook runs consistently on every render.
  // --------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    getBackendHealth()
      .then((data) => {
        if (cancelled) return;

        setBackendStatus(
          data.status === "healthy" ? "Connected" : "Unavailable"
        );

        setBackendMessage(
          data.message || "FastAPI is responding."
        );
        setTeacherAvailable(
          data.online_provider_available === true || data.teacher_available === true,
        );
        setYoutubeAvailable(data.youtube_available === true);
        setQuantaLoaded(Boolean(data.quantaLM_loaded));
      })
      .catch((error) => {
        if (cancelled) return;

        setBackendStatus("Disconnected");
        setBackendMessage(error.message);
        setTeacherAvailable(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (authStatus !== "authenticated") return undefined;

    let cancelled = false;
    const verifySession = () => {
      getCurrentUser()
        .then((authenticatedUser) => {
          if (!cancelled) setUser(authenticatedUser);
        })
        .catch((error) => {
          if (cancelled) return;
          if (error.status === 401) {
            handleAuthenticationExpired();
          } else {
            setNotice(error.message || "Unable to verify your session.");
          }
        });
    };

    const intervalId = window.setInterval(verifySession, 60_000);
    window.addEventListener("focus", verifySession);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      window.removeEventListener("focus", verifySession);
    };
  }, [authStatus, handleAuthenticationExpired]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !user?.id) return undefined;

    let cancelled = false;
    const requestId = historyRequestId.current + 1;
    historyRequestId.current = requestId;
    listConversations()
      .then(({ conversations: loadedConversations }) => {
        if (!cancelled && historyRequestId.current === requestId) {
          setConversations(loadedConversations);
        }
      })
      .catch((error) => {
        if (cancelled || historyRequestId.current !== requestId) return;
        if (error.status === 401) {
          handleAuthenticationExpired();
        } else {
          setHistoryError(
            error.message || "Unable to load your conversations.",
          );
        }
      })
      .finally(() => {
        if (!cancelled && historyRequestId.current === requestId) {
          setHistoryLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [authStatus, user?.id, handleAuthenticationExpired]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !user?.id) return undefined;

    let cancelled = false;
    listPlannerTasks()
      .then((loadedTasks) => {
        if (!cancelled) {
          setTasks((currentTasks) => {
            const tasksById = new Map(
              loadedTasks.map((task) => [task.id, task]),
            );
            for (const task of currentTasks) {
              if (!tasksById.has(task.id)) tasksById.set(task.id, task);
            }
            return sortPlannerTasks([...tasksById.values()]);
          });
        }
      })
      .catch((error) => {
        if (cancelled) return;
        if (error.status === 401) handleAuthenticationExpired();
        else setPlannerError(error.message || "Unable to load planner tasks.");
      })
      .finally(() => {
        if (!cancelled) setPlannerLoading(false);
      });

    getLearningProgress()
      .then((progress) => {
        if (!cancelled) {
          setLearningProgress(progress);
          setProgressError("");
        }
      })
      .catch((error) => {
        if (cancelled) return;
        if (error.status === 401) handleAuthenticationExpired();
        else setProgressError(error.message || "Unable to load learning progress.");
      });

    return () => {
      cancelled = true;
    };
  }, [authStatus, user?.id, handleAuthenticationExpired]);

  useEffect(() => {
    if (!user?.id || tasks.length === 0) return undefined;

    const checkReminders = () => {
      try {
        const dueTasks = takeDuePlannerReminders(user.id, tasks);
        if (dueTasks.length === 0) return;
        setReminderAlerts((current) => {
          const known = new Set(current.map((task) => `${task.id}|${task.due_date}`));
          return [
            ...current,
            ...dueTasks.filter((task) => !known.has(`${task.id}|${task.due_date}`)),
          ];
        });
        if (
          browserNotificationsEnabled &&
          typeof Notification !== "undefined" &&
          Notification.permission === "granted"
        ) {
          for (const task of dueTasks) {
            new Notification(`QuantaAI reminder: ${task.title}`, {
              body: task.description || `Due today: ${task.due_date}`,
              tag: `planner-task-${task.id}-${task.due_date}`,
            });
          }
        }
      } catch (error) {
        setPlannerReminderError(
          `Planner reminders could not be checked. ${error.message}`,
        );
      }
    };

    checkReminders();
    const reminderInterval = window.setInterval(checkReminders, 30_000);
    return () => window.clearInterval(reminderInterval);
  }, [browserNotificationsEnabled, tasks, user?.id]);

  useEffect(() => {
    let cancelled = false;

    getCurrentUser()
      .then((authenticatedUser) => {
        if (cancelled) return;
        setUser(authenticatedUser);
        loadReminderSettings(authenticatedUser.id);
        setPlannerLoading(true);
        setAuthStatus("authenticated");
        setCurrentScreen("dashboard");
        setActiveAgent(surfaceAssistant.id);
        setSelectedAgents([]);
        setCollaborationMode(false);
        setSidebarCollapsed(true);
      })
      .catch(() => {
        if (cancelled) return;
        setUser(null);
        setAuthStatus("unauthenticated");
        setMessages([]);
        setTasks([]);
        setPlannerReminderTimes({});
        setPlannerLoading(false);
        setPlannerError("");
        setLearningProgress(null);
        setProgressError("");
        setStudyDocuments([]);
        setSelectedStudyDocumentId(null);
        setStudyDocumentsLoading(false);
        setStudyDocumentError("");
        setSurfaceAttachments([]);
        setSurfaceDocumentUploading(false);
        setSurfaceDocumentProgress(0);
        setSurfaceUploadingFilename("");
        setSurfaceDocumentError("");
        setConversations([]);
        setActiveConversationId(null);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // --------------------------------------------------
  // 3. CHAT FUNCTION
  // --------------------------------------------------

  async function openConversation(conversationId) {
    if (busy || conversationLoading || surfaceDocumentUploading) return;

    const requestId = selectionRequestId.current + 1;
    selectionRequestId.current = requestId;
    setConversationLoading(true);
    setHistoryError("");

    try {
      const conversation = await getConversation(conversationId);
      if (selectionRequestId.current !== requestId) return;
      setActiveConversationId(conversation.id);
      setMessages(conversation.messages.map(formatConversationMessage));
      setSelectedStudyDocumentId(null);
      setSurfaceDocumentError("");
      const lastSurfaceMessage = [...conversation.messages]
        .reverse()
        .find((item) => item.role === "user" && item.agent === "Surface");
      setSurfaceAttachments(lastSurfaceMessage?.document_attachments || []);
      const conversationAgents = [
        ...new Set(
          conversation.messages
            .filter((item) => item.role === "assistant" && item.agent)
            .map((item) => item.agent)
            .filter(
              (agentId) => agents.some((agent) => agent.id === agentId),
            ),
        ),
      ];
      const latestAssistantAgent = [...conversation.messages]
        .reverse()
        .find((item) => item.role === "assistant")?.agent;
      if (latestAssistantAgent === surfaceAssistant.id) {
        setActiveAgent(surfaceAssistant.id);
        setSelectedAgents([]);
        setCollaborationMode(false);
      } else if (conversationAgents.length > 0) {
        if (
          conversationAgents.length === 1 &&
          conversationAgents[0] === "Study"
        ) {
          setStudyDocumentsLoading(true);
          setStudyDocumentError("");
        }
        setActiveAgent(latestAssistantAgent || conversationAgents.at(-1));
        setCollaborationMode(conversationAgents.length > 1);
        setSelectedAgents(
          conversationAgents.length > 1 ? conversationAgents : [],
        );
      } else {
        setActiveAgent(surfaceAssistant.id);
        setSelectedAgents([]);
        setCollaborationMode(false);
      }
      setRenamingConversationId(null);
      setConfirmingDeleteId(null);
      setNotice(`Opened ${conversation.title}`);
    } catch (error) {
      if (selectionRequestId.current !== requestId) return;
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setHistoryError(error.message || "Unable to open this conversation.");
      }
    } finally {
      if (selectionRequestId.current === requestId) {
        setConversationLoading(false);
      }
    }
  }

  async function handleStudyPdfUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setStudyDocumentError("");
    if (
      !file.name.toLowerCase().endsWith(".pdf") ||
      (file.type && file.type !== "application/pdf")
    ) {
      setStudyDocumentError("Select a PDF file to upload.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setStudyDocumentError("PDF files must be 10 MB or smaller.");
      return;
    }

    setStudyDocumentUploading(true);
    setStudyDocumentProgress(0);
    try {
      const document = await uploadStudyDocument(file, setStudyDocumentProgress);
      setStudyDocuments((previous) => [
        document,
        ...previous.filter((item) => item.id !== document.id),
      ]);
      if (document.status === "indexed") {
        setSelectedStudyDocumentId(document.id);
        setStudyDocumentMenuOpen(false);
        setStudyDocumentPickerOpen(false);
        setNotice(`${document.filename} is indexed and selected`);
      } else {
        setSelectedStudyDocumentId(null);
        setStudyDocumentError(
          document.status_message || `${document.filename} could not be indexed.`,
        );
        setNotice(`${document.filename} could not be indexed`);
      }
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setStudyDocumentError(
          error.message || "Unable to upload this PDF. Please try again.",
        );
      }
    } finally {
      setStudyDocumentUploading(false);
      setStudyDocumentProgress(0);
    }
  }

  async function handleSurfacePdfUpload(event) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (files.length === 0) return;

    const invalidFile = files.find(
      (file) =>
        !file.name.toLowerCase().endsWith(".pdf") ||
        (file.type && file.type !== "application/pdf") ||
        file.size > 10 * 1024 * 1024,
    );
    if (invalidFile) {
      setSurfaceDocumentError(
        invalidFile.size > 10 * 1024 * 1024
          ? `${invalidFile.name} is larger than 10 MB.`
          : `${invalidFile.name} is not a valid PDF selection.`,
      );
      return;
    }
    if (files.length + surfaceAttachments.length > 5) {
      setSurfaceDocumentError("Attach no more than five PDFs to one conversation.");
      return;
    }

    setSurfaceDocumentError("");
    setSurfaceDocumentUploading(true);
    try {
      for (const file of files) {
        setSurfaceUploadingFilename(file.name);
        setSurfaceDocumentProgress(0);
        const document = await uploadStudyDocument(
          file,
          setSurfaceDocumentProgress,
        );
        if (document.status !== "indexed") {
          throw new Error(
            document.status_message || `${document.filename} could not be indexed.`,
          );
        }
        setStudyDocuments((previous) => [
          document,
          ...previous.filter((item) => item.id !== document.id),
        ]);
        setSurfaceAttachments((previous) =>
          previous.some((item) => item.id === document.id)
            ? previous
            : [...previous, { id: document.id, filename: document.filename }],
        );
      }
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setSurfaceDocumentError(
          error.message || "Unable to upload this PDF. Please try again.",
        );
      }
    } finally {
      setSurfaceDocumentUploading(false);
      setSurfaceDocumentProgress(0);
      setSurfaceUploadingFilename("");
    }
  }

  async function removeStudyDocument(document) {
    setStudyDocumentMutationId(document.id);
    setStudyDocumentError("");
    try {
      await deleteStudyDocument(document.id);
      setStudyDocuments((previous) =>
        previous.filter((item) => item.id !== document.id),
      );
      setStudyDocumentDeleteConfirmationId(null);
      setSelectedStudyDocumentId((current) =>
        current === document.id ? null : current,
      );
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setStudyDocumentError(
          error.message || "Unable to remove this PDF. Please try again.",
        );
      }
    } finally {
      setStudyDocumentMutationId(null);
    }
  }

  async function sendMessage(text = message) {
    const cleanText = text.trim();
    const requestId = selectionRequestId.current;

    if (
      !cleanText ||
      busy ||
      surfaceDocumentUploading ||
      historyLoading ||
      conversationLoading ||
      (collaborationMode && selectedAgents.length === 0)
    ) {
      return;
    }

    setStudyDocumentMenuOpen(false);
    setStudyDocumentPickerOpen(false);

    const requestAgents = activeParticipants;
    const requestSurfaceAttachments =
      activeAgent === surfaceAssistant.id ? surfaceAttachments : [];
    const requestMode = chatMode;
    const conversationHistory = messages
      .filter((item) => item.role === "user" || item.role === "assistant")
      .slice(-12)
      .map((item) => ({
        role: item.role,
        content: item.text.slice(0, 4000),
      }));
    setBusy(true);
    setRespondingAgents(requestAgents);
    setSearchingAgents(
      shouldSearchOnline(cleanText, requestMode) ? requestAgents : [],
    );
    setHistoryError("");
    setNotice(
      `Contacting ${requestAgents.map((agent) => agent.name).join(", ")} · ${requestMode.toUpperCase()}`,
    );

    try {
      let conversationId = activeConversationId;
      if (conversationId === null) {
        const created = await createConversation(titleFromMessage(cleanText));
        if (selectionRequestId.current !== requestId) return;
        conversationId = created.id;
        setConversations((previous) => [
          created,
          ...previous.filter((item) => item.id !== created.id),
        ]);
        setActiveConversationId(created.id);
      } else if (messages.length === 0) {
        const activeConversation = conversations.find(
          (conversation) => conversation.id === conversationId,
        );
        if (activeConversation?.title === "New chat") {
          try {
            const renamed = await renameConversationRequest(
              conversationId,
              titleFromMessage(cleanText),
            );
            if (selectionRequestId.current !== requestId) return;
            setConversations((previous) =>
              previous.map((conversation) =>
                conversation.id === conversationId ? renamed : conversation,
              ),
            );
          } catch (error) {
            if (selectionRequestId.current !== requestId) return;
            if (error.status === 401) {
              handleAuthenticationExpired();
              return;
            }
            setHistoryError(
              `The message will be saved, but its title could not be updated: ${error.message}`,
            );
          }
        }
      }

      const savedUserMessage = await appendConversationMessages(
        conversationId,
        [
          {
            role: "user",
            content: cleanText,
            agent: requestAgents[0].id,
            ...(requestAgents[0].id === surfaceAssistant.id
              ? { document_attachments: requestSurfaceAttachments }
              : {}),
          },
        ],
      );
      if (selectionRequestId.current !== requestId) return;
      setMessages((previous) => [
        ...previous,
        ...savedUserMessage.messages.map(formatConversationMessage),
      ]);
      setMessage("");

      const outcomes = await Promise.all(
        requestAgents.map(async (agent) => {
          try {
            const result = await sendChatMessage({
              message: cleanText,
              agent: agent.id,
              mode: requestMode,
              tutorMode,
              conversationHistory,
              documentId:
                !collaborationMode &&
                agent.id === "Study"
                  ? selectedStudyDocumentId
                  : undefined,
              conversationId:
                agent.id === surfaceAssistant.id ? conversationId : undefined,
            });

            if (typeof result.reply !== "string" || !result.reply.trim()) {
              throw new Error("The backend returned an empty reply.");
            }

            return {
              agent,
              result,
              timestamp: new Date().toISOString(),
            };
          } catch (error) {
            return { agent, error, timestamp: new Date().toISOString() };
          }
        }),
      );

      if (selectionRequestId.current !== requestId) return;
      if (outcomes.some((outcome) => outcome.error?.status === 401)) {
        handleAuthenticationExpired();
        return;
      }

      const successfulOutcomes = outcomes.filter((outcome) => outcome.result);
      const scheduleNotices = new Map();
      for (const outcome of outcomes) {
        if (
          outcome.agent.id !== "Planner" ||
          !outcome.result ||
          (!outcome.result.schedule_tasks?.length &&
            !outcome.result.schedule_tasks_error)
        ) continue;

        const notices = [];
        let result = { created: [], duplicates: [], failures: [] };
        try {
          if (outcome.result.schedule_tasks?.length) {
            result = await createPlannerSchedule(
              outcome.result.schedule_tasks,
              (created) =>
                setTasks((previous) =>
                  sortPlannerTasks([
                    ...previous.filter((task) => task.id !== created.id),
                    created,
                  ]),
                ),
            );
          }
          const createdCount = result.created.length;
          const scheduledDates = [
            ...result.created,
            ...result.duplicates,
          ]
            .map((task) => task.due_date)
            .sort();
          const dateRange = scheduledDates.length
            ? ` Dates covered: ${formatPlannerDate(scheduledDates[0])}${
                scheduledDates[0] === scheduledDates.at(-1)
                  ? ""
                  : ` – ${formatPlannerDate(scheduledDates.at(-1))}`
              }.`
            : "";

          if (createdCount > 0) {
            notices.push(
              `Added ${createdCount} task${createdCount === 1 ? "" : "s"} to Planner & To-do.${dateRange}`,
            );
          } else if (result.duplicates.length > 0) {
            notices.push(
              `No new tasks were created; ${result.duplicates.length} matching task${result.duplicates.length === 1 ? " is" : "s are"} already in Planner & To-do.${dateRange}`,
            );
          }
          if (result.duplicates.length > 0 && createdCount > 0) {
            notices.push(
              `Skipped ${result.duplicates.length} duplicate task${result.duplicates.length === 1 ? "" : "s"}.`,
            );
          }
          if (result.failures.length > 0) {
            notices.push(
              `Could not save ${result.failures.length} task${result.failures.length === 1 ? "" : "s"}. ${
                createdCount > 0
                  ? `${createdCount} task${createdCount === 1 ? " was" : "s were"} saved successfully.`
                  : "No tasks were saved."
              } ${result.failures[0].message}`,
            );
          }
        } catch (error) {
          if (error.status === 401) {
            handleAuthenticationExpired();
            return;
          }
          notices.push(
            `Could not check existing tasks, so no schedule tasks were added. ${error.message}`,
          );
        }
        if (outcome.result.schedule_tasks_error) {
          notices.push(outcome.result.schedule_tasks_error);
        }
        if (notices.length > 0) {
          scheduleNotices.set(outcome.agent.id, notices.join(" "));
        }
      }

      const responseRecords = outcomes.map(({ agent, result, error }) => ({
        role: error ? "error" : "assistant",
        content: error
          ? error.message || `${agent.name} could not respond. Please try again.`
          : [
              result.reply,
              scheduleNotices.get(agent.id),
            ]
              .filter(Boolean)
              .join("\n\n"),
        agent: agent.id,
        model: result?.model || null,
        provider: result?.provider || null,
        experimental: result?.experimental === true,
        reliability_warning: result?.reliability_warning === true,
        videos: result?.videos || [],
        video_message: result?.video_message || "",
        search_performed: result?.search_performed === true,
        search_agent: result?.search_agent || null,
        search_message: result?.search_message || "",
        sources: result?.sources || [],
        document_source: result?.document_source || null,
        document_pages: result?.document_pages || [],
      }));
      if (collaborationMode && requestAgents.length > 1 && successfulOutcomes.length > 0) {
        setRespondingAgents([
          ...requestAgents,
          {
            id: "Collaboration",
            name: "Collaboration Summary",
            icon: "✣",
          },
        ]);
        let synthesisRecord;
        try {
          const synthesis = await synthesizeCollaboration({
            message: cleanText,
            contributions: successfulOutcomes.map(({ agent, result }) => ({
              agent: agent.id,
              response: result.reply,
            })),
            failedAgents: outcomes
              .filter((outcome) => outcome.error)
              .map(({ agent }) => agent.id),
            mode: requestMode,
            conversationHistory,
          });
          if (selectionRequestId.current !== requestId) return;
          if (typeof synthesis.reply !== "string" || !synthesis.reply.trim()) {
            throw new Error("The backend returned an empty summary.");
          }
          synthesisRecord = {
            role: "assistant",
            content: synthesis.reply,
            agent: "Collaboration",
            model: synthesis.model,
            provider: synthesis.provider,
          };
        } catch (error) {
          if (selectionRequestId.current !== requestId) return;
          if (error.status === 401) {
            handleAuthenticationExpired();
            return;
          }
          synthesisRecord = {
            role: "error",
            content: `The agent responses are available, but a combined summary could not be generated: ${error.message || "Please try again."}`,
            agent: "Collaboration",
          };
        }
        responseRecords.push(synthesisRecord);
      }

      let displayedResponses;
      try {
        const savedResponses = await appendConversationMessages(
          conversationId,
          responseRecords,
        );
        if (selectionRequestId.current !== requestId) return;
        displayedResponses =
          savedResponses.messages.map((savedMessage, index) => ({
            ...formatConversationMessage(savedMessage),
            videos: responseRecords[index]?.videos || [],
            videoMessage: responseRecords[index]?.video_message || "",
            searchMessage: responseRecords[index]?.search_message || "",
          }));
      } catch (error) {
        if (selectionRequestId.current !== requestId) return;
        if (error.status === 401) {
          handleAuthenticationExpired();
          return;
        }
        displayedResponses = responseRecords.map((record, index) => ({
          ...formatConversationMessage({
            ...record,
            id: `unsaved-${Date.now()}-${index}`,
            created_at: new Date().toISOString(),
          }),
          videos: record.videos,
          videoMessage: record.video_message,
          searchMessage: record.search_message,
        }));
        setHistoryError(
          `The reply was generated but could not be saved: ${error.message}`,
        );
      }
      setMessages((previous) => [...previous, ...displayedResponses]);

      if (successfulOutcomes.some(({ result }) => result.model === "QuantaLM")) {
        setQuantaLoaded(true);
      }
      const providersUsed = [
        ...new Set(
          successfulOutcomes
            .map(({ result }) => result.provider)
            .filter(Boolean),
        ),
      ].join(", ");
      setNotice(
        successfulOutcomes.length === outcomes.length
          ? `${requestAgents.length === 1 ? requestAgents[0].name : `${successfulOutcomes.length} agents`} responded · ${providersUsed || requestMode.toUpperCase()}`
          : `${successfulOutcomes.length} of ${outcomes.length} agents responded`,
      );
    } catch (error) {
      if (selectionRequestId.current !== requestId) return;
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setHistoryError(error.message || "Unable to save this conversation.");
        setNotice("Message could not be saved");
      }
    } finally {
      if (selectionRequestId.current === requestId) {
        setRespondingAgents([]);
        setSearchingAgents([]);
        setBusy(false);
      }
    }
  }

  function toggleAgent(agentId) {
    if (busy) return;
    setStudyDocumentMenuOpen(false);
    setStudyDocumentPickerOpen(false);

    if (!collaborationMode) {
      if (agentId === "Study" && activeAgent !== "Study") {
        setStudyDocumentsLoading(true);
        setStudyDocumentError("");
      }
      setActiveAgent(agentId);
      setSelectedAgents([]);
      return;
    }

    if (selectedAgents.includes(agentId)) {
      const nextAgents = selectedAgents.filter(
        (selectedId) => selectedId !== agentId,
      );
      setSelectedAgents(nextAgents);
      if (activeAgent === agentId) {
        setActiveAgent(nextAgents.at(-1) || surfaceAssistant.id);
      }
      return;
    }

    setSelectedAgents([...selectedAgents, agentId]);
    setActiveAgent(agentId);
  }

  function selectSurfaceAssistant() {
    setStudyDocumentMenuOpen(false);
    setStudyDocumentPickerOpen(false);
    setActiveAgent(surfaceAssistant.id);
    setSelectedAgents([]);
    setCollaborationMode(false);
  }

  function toggleCollaboration() {
    setStudyDocumentMenuOpen(false);
    setStudyDocumentPickerOpen(false);
    if (collaborationMode) {
      if (activeAgent === "Study") {
        setStudyDocumentsLoading(true);
        setStudyDocumentError("");
      }
      setCollaborationMode(false);
      setSelectedAgents([]);
      return;
    }
    setCollaborationMode(true);
    setSelectedAgents(
      activeAgent === surfaceAssistant.id ? [] : [activeAgent],
    );
  }

  async function startNewChat() {
    if (busy || historyLoading || conversationLoading || surfaceDocumentUploading) return;
    const requestId = selectionRequestId.current;
    setConversationMutation("new");
    setHistoryError("");
    try {
      const conversation = await createConversation();
      if (selectionRequestId.current !== requestId) return;
      setConversations((previous) => [
        conversation,
        ...previous.filter((item) => item.id !== conversation.id),
      ]);
      selectionRequestId.current += 1;
      setActiveConversationId(conversation.id);
      setMessages([]);
      setMessage("");
      setSelectedStudyDocumentId(null);
      setSurfaceAttachments([]);
      setSurfaceDocumentError("");
      setRenamingConversationId(null);
      setConfirmingDeleteId(null);
      setNotice("New conversation started");
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setHistoryError(error.message || "Unable to create a new conversation.");
      }
    } finally {
      setConversationMutation("");
    }
  }

  async function saveConversationTitle(conversationId) {
    const title = renameDraft.trim();
    if (!title) {
      setHistoryError("Conversation titles cannot be empty.");
      return;
    }
    setConversationMutation(String(conversationId));
    setHistoryError("");
    try {
      const updated = await renameConversationRequest(conversationId, title);
      setConversations((previous) =>
        previous.map((item) => (item.id === conversationId ? updated : item)),
      );
      setRenamingConversationId(null);
      setRenameDraft("");
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setHistoryError(error.message || "Unable to rename this conversation.");
      }
    } finally {
      setConversationMutation("");
    }
  }

  async function removeConversation(conversationId) {
    setConversationMutation(String(conversationId));
    setHistoryError("");
    try {
      await deleteConversationRequest(conversationId);
      setConversations((previous) =>
        previous.filter((item) => item.id !== conversationId),
      );
      if (activeConversationId === conversationId) {
        selectionRequestId.current += 1;
        setActiveConversationId(null);
        setMessages([]);
        setMessage("");
      }
      setConfirmingDeleteId(null);
      setRenamingConversationId(null);
    } catch (error) {
      if (error.status === 401) {
        handleAuthenticationExpired();
      } else {
        setHistoryError(error.message || "Unable to delete this conversation.");
      }
    } finally {
      setConversationMutation("");
    }
  }

  // --------------------------------------------------
  // 4. TASK MANAGER FUNCTIONS
  // --------------------------------------------------

  function loadReminderSettings(userId) {
    try {
      setPlannerReminderTimes(loadPlannerReminderTimes(userId));
      setPlannerReminderError("");
    } catch (error) {
      setPlannerReminderError(
        `Reminder settings could not be loaded from this browser. ${error.message}`,
      );
    }
  }

  async function handleCreateTask(task) {
    setPlannerError("");
    setTaskMutation("new");
    try {
      const created = await createPlannerTask(task);
      setTasks((previous) => sortPlannerTasks([...previous, created]));
      setShowTasks(false);
      return true;
    } catch (error) {
      if (error.status === 401) handleAuthenticationExpired();
      else setPlannerError(error.message || "Unable to create this task.");
      return false;
    } finally {
      setTaskMutation("");
    }
  }

  async function handleUpdateTask(taskId, updates) {
    setPlannerError("");
    setTaskMutation(String(taskId));
    try {
      const updated = await updatePlannerTask(taskId, updates);
      setTasks((previous) =>
        sortPlannerTasks(
          previous.map((task) => (task.id === taskId ? updated : task)),
        ),
      );
      if (updated.completed) {
        setReminderAlerts((previous) =>
          previous.filter((task) => task.id !== taskId),
        );
      }
      return true;
    } catch (error) {
      if (error.status === 401) handleAuthenticationExpired();
      else setPlannerError(error.message || "Unable to update this task.");
      return false;
    } finally {
      setTaskMutation("");
    }
  }

  async function handleDeleteTask(taskId) {
    setPlannerError("");
    setTaskMutation(String(taskId));
    try {
      await deletePlannerTask(taskId);
      setTasks((previous) => previous.filter((task) => task.id !== taskId));
      setReminderAlerts((previous) =>
        previous.filter((task) => task.id !== taskId),
      );
      return true;
    } catch (error) {
      if (error.status === 401) handleAuthenticationExpired();
      else setPlannerError(error.message || "Unable to delete this task.");
      return false;
    } finally {
      setTaskMutation("");
    }
  }

  function handleReminderTimeChange(taskId, time) {
    try {
      setPlannerReminderTimes(
        savePlannerReminderTime(user.id, taskId, time),
      );
      setPlannerReminderError("");
    } catch (error) {
      setPlannerReminderError(
        `Reminder time could not be saved. ${error.message}`,
      );
    }
  }

  async function enableBrowserNotifications() {
    if (typeof Notification === "undefined") {
      setPlannerReminderError(
        "Browser notifications are not supported here. In-app reminders still work while QuantaAI is open.",
      );
      return;
    }
    try {
      const permission = await Notification.requestPermission();
      setBrowserNotificationsEnabled(permission === "granted");
      setPlannerReminderError(
        permission === "granted"
          ? ""
          : "Browser notification permission was not granted. In-app reminders remain available while QuantaAI is open.",
      );
    } catch (error) {
      setPlannerReminderError(
        `Browser notification permission could not be requested. ${error.message}`,
      );
    }
  }

  async function addTask(event) {
    event.preventDefault();
    if (!taskTitle.trim() || !taskDate) return;
    const saved = await handleCreateTask({
      title: taskTitle.trim(),
      due_date: taskDate,
    });
    if (saved) {
      setTaskTitle("");
      setTaskDate("");
    }
  }

  async function handleLogin(credentials) {
    const result = await loginUser(credentials);
    selectionRequestId.current += 1;
    historyRequestId.current += 1;
    setUser(result.user);
    setReminderAlerts([]);
    loadReminderSettings(result.user.id);
    setStudyDocuments([]);
    setSelectedStudyDocumentId(null);
    setStudyDocumentsLoading(false);
    setStudyDocumentError("");
    setStudyDocumentUploading(false);
    setStudyDocumentProgress(0);
    setStudyDocumentMutationId(null);
    setSurfaceAttachments([]);
    setSurfaceDocumentUploading(false);
    setSurfaceDocumentProgress(0);
    setSurfaceUploadingFilename("");
    setSurfaceDocumentError("");
    setActiveAgent(surfaceAssistant.id);
    setSelectedAgents([]);
    setCollaborationMode(false);
    setSidebarCollapsed(true);
    setConversations([]);
    setActiveConversationId(null);
    setMessages([]);
    setConversationLoading(false);
    setConversationMutation("");
    setBusy(false);
    setRespondingAgents([]);
    setHistoryLoading(true);
    setHistoryError("");
    setPlannerLoading(true);
    setAuthStatus("authenticated");
    setCurrentScreen("dashboard");
    setActivePage("workspace");
    setAccountMenuOpen(false);
    setNotice(`Welcome back, ${result.user.name}.`);
  }

  async function handleSignup(account) {
    const result = await registerUser(account);
    selectionRequestId.current += 1;
    historyRequestId.current += 1;
    setUser(result.user);
    setReminderAlerts([]);
    loadReminderSettings(result.user.id);
    setStudyDocuments([]);
    setSelectedStudyDocumentId(null);
    setStudyDocumentsLoading(false);
    setStudyDocumentError("");
    setStudyDocumentUploading(false);
    setStudyDocumentProgress(0);
    setStudyDocumentMutationId(null);
    setSurfaceAttachments([]);
    setSurfaceDocumentUploading(false);
    setSurfaceDocumentProgress(0);
    setSurfaceUploadingFilename("");
    setSurfaceDocumentError("");
    setActiveAgent(surfaceAssistant.id);
    setSelectedAgents([]);
    setCollaborationMode(false);
    setSidebarCollapsed(true);
    setConversations([]);
    setActiveConversationId(null);
    setMessages([]);
    setConversationLoading(false);
    setConversationMutation("");
    setBusy(false);
    setRespondingAgents([]);
    setHistoryLoading(true);
    setHistoryError("");
    setPlannerLoading(true);
    setAuthStatus("authenticated");
    setCurrentScreen("dashboard");
    setActivePage("workspace");
    setAccountMenuOpen(false);
    setNotice(`Welcome to QuantaAI, ${result.user.name}.`);
  }

  async function handleLogout() {
    await logoutUser();
    selectionRequestId.current += 1;
    historyRequestId.current += 1;
    setUser(null);
    setReminderAlerts([]);
    setPlannerReminderTimes({});
    setAuthStatus("unauthenticated");
    setCurrentScreen("login");
    setStudyDocuments([]);
    setSelectedStudyDocumentId(null);
    setStudyDocumentsLoading(false);
    setStudyDocumentError("");
    setStudyDocumentUploading(false);
    setStudyDocumentProgress(0);
    setStudyDocumentMutationId(null);
    setSurfaceAttachments([]);
    setSurfaceDocumentUploading(false);
    setSurfaceDocumentProgress(0);
    setSurfaceUploadingFilename("");
    setSurfaceDocumentError("");
    setAccountMenuOpen(false);
    setAccountMenuError("");
    setActiveAgent(surfaceAssistant.id);
    setSelectedAgents([]);
    setCollaborationMode(false);
    setSidebarCollapsed(true);
    setMessages([]);
    setConversations([]);
    setActiveConversationId(null);
    setConversationLoading(false);
    setConversationMutation("");
    setBusy(false);
    setRespondingAgents([]);
    setHistoryLoading(false);
    setHistoryError("");
    setTasks([]);
    setPlannerLoading(false);
    setPlannerError("");
    setTaskMutation("");
    setLearningProgress(null);
    setProgressError("");
    setNotice("You have signed out.");
  }

  async function handleProfileUpdate(profile) {
    try {
      const updatedUser = await updateCurrentUser(profile);
      setUser(updatedUser);
      setNotice("Your profile has been updated.");
      return updatedUser;
    } catch (error) {
      if (error.status === 401) handleAuthenticationExpired();
      throw error;
    }
  }

  function navigateTo(page) {
    setActivePage(page);
    setAccountMenuOpen(false);
    setAccountMenuError("");
  }

  // --------------------------------------------------
  // 5. LOGIN SCREEN
  // --------------------------------------------------

  if (authStatus === "checking") {
    return (
      <main className="auth-loading" role="status">
        Checking your QuantaAI session...
      </main>
    );
  }

  if (authStatus !== "authenticated" && currentScreen === "login") {
    return (
      <Login
        onLogin={handleLogin}
        onSignup={() => setCurrentScreen("signup")}
      />
    );
  }

  // --------------------------------------------------
  // 6. SIGNUP SCREEN
  // --------------------------------------------------

  if (authStatus !== "authenticated" && currentScreen === "signup") {
    return (
      <Signup
        onSignup={handleSignup}
        onBackToLogin={() => setCurrentScreen("login")}
      />
    );
  }

  if (authStatus !== "authenticated" || !user) {
    return (
      <Login
        onLogin={handleLogin}
        onSignup={() => setCurrentScreen("signup")}
      />
    );
  }

  // --------------------------------------------------
  // 7. TEMPORARY DASHBOARD
  // --------------------------------------------------

  return (
    <div
      className={`app-shell ${sidebarCollapsed ? "sidebar-is-collapsed" : ""} ${
        activePage === "workspace" ? "" : "page-without-dashboard"
      }`}
      data-theme={theme}
    >
      {reminderAlerts.length > 0 && (
        <aside className="planner-reminder-alerts" aria-label="Due planner reminders">
          {reminderAlerts.map((task) => (
            <div
              className="planner-reminder-alert"
              role="status"
              key={`${task.id}|${task.due_date}`}
            >
              <span>
                <strong>Task reminder</strong>
                {task.title} is due today ({task.due_date}).
              </span>
              <button
                type="button"
                onClick={() =>
                  setReminderAlerts((previous) =>
                    previous.filter(
                      (item) =>
                        item.id !== task.id || item.due_date !== task.due_date,
                    ),
                  )
                }
                aria-label={`Dismiss reminder for ${task.title}`}
              >
                Dismiss
              </button>
              <button type="button" onClick={() => navigateTo("planner")}>
                Open Planner
              </button>
            </div>
          ))}
        </aside>
      )}
      <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <div className="brand">
          <button
            className="brand-mark"
            type="button"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            aria-label={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
            aria-expanded={!sidebarCollapsed}
            title={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
          >
            <img src="/quantaai-icon.png" alt="QuantaAI logo" />
          </button>
          <div className="brand-copy">
            <h1>QuantaAI</h1>
            <span>YOUR LEARNING SPACE</span>
          </div>
        </div>

        <div className="sidebar-primary-actions">
          <button
            className="new-chat-button"
            type="button"
            onClick={() => startNewChat()}
            disabled={
              busy ||
              historyLoading ||
              conversationLoading ||
              conversationMutation !== ""
            }
            aria-label="Start a new chat"
            title="New chat"
          >
            <Plus size={16} aria-hidden="true" />
            <span>New chat</span>
          </button>

          <button
            className={`surface-button ${activeAgent === surfaceAssistant.id ? "selected" : ""}`}
            type="button"
            onClick={selectSurfaceAssistant}
            disabled={busy}
            aria-pressed={activeAgent === surfaceAssistant.id}
            aria-label="Surface AI"
            title="Surface AI"
          >
            <surfaceAssistant.icon size={16} aria-hidden="true" />
            <span>Surface AI</span>
          </button>
        </div>

        <div className="sidebar-specialist-heading">
          <div className="sidebar-label">SPECIALIST AGENTS</div>
          <button
            className={`collaboration-toggle ${collaborationMode ? "enabled" : ""}`}
            type="button"
            onClick={toggleCollaboration}
            disabled={busy}
            aria-pressed={collaborationMode}
            aria-label={collaborationMode ? "Disable collaboration mode" : "Enable collaboration mode"}
            title={collaborationMode ? "Disable collaboration mode" : "Enable collaboration mode"}
          >
            <UsersRound size={15} aria-hidden="true" />
            <span>{collaborationMode ? "Collaboration on" : "Collaborate with agents"}</span>
            <span className="collaboration-state">{collaborationMode ? "ON" : "OFF"}</span>
          </button>
        </div>

        <nav className="agent-list" aria-label="Select a specialist agent">
          {agents.map((agent) => (
            (() => {
              const isSelected = collaborationMode
                ? selectedAgents.includes(agent.id)
                : activeAgent === agent.id;
              return (
                <button
                  key={agent.id}
                  type="button"
                  className={`agent-button ${isSelected ? "selected" : ""}`}
                  onClick={() => toggleAgent(agent.id)}
                  disabled={busy}
                  aria-label={agent.name}
                  aria-pressed={isSelected}
                  title={agent.name}
                >
                  <span className="agent-icon">
                    <agent.icon size={17} strokeWidth={1.8} aria-hidden="true" />
                  </span>
                  <span className="agent-copy">
                    <strong>{agent.name}</strong>
                    <small>{agent.desc}</small>
                  </span>
                  {isSelected && (
                    <span className="agent-selected-mark" aria-hidden="true">
                      <Check size={12} />
                    </span>
                  )}
                </button>
              );
            })()
          ))}
        </nav>

          <nav className="sidebar-tools" aria-label="Learning tools">
            {[
              { id: "planner", icon: CalendarDays, label: "Planner" },
              { id: "quiz", icon: ClipboardCheck, label: "Quizzes" },
              { id: "flashcards", icon: FileText, label: "Flashcards" },
              { id: "documents", icon: FolderOpen, label: "My Uploads" },
              { id: "progress", icon: ListChecks, label: "Progress" },
            ].map((tool) => (
              <button
                className={`sidebar-tool-button ${
                  activePage === tool.id ? "selected" : ""
                }`}
                type="button"
                key={tool.id}
                onClick={() => navigateTo(tool.id)}
                aria-current={activePage === tool.id ? "page" : undefined}
                title={tool.label}
              >
                <tool.icon size={16} aria-hidden="true" />
                <span>{tool.label}</span>
              </button>
            ))}
          </nav>

          <section className="conversation-history" aria-label="Conversation history">
            <div className="recent-chats-controls">
              <button
                className={`recent-chats-toggle ${recentChatsVisible ? "expanded" : ""}`}
                type="button"
                onClick={() => {
                  if (sidebarCollapsed) {
                    setSidebarCollapsed(false);
                    setRecentChatsExpanded(true);
                    return;
                  }
                  setRecentChatsExpanded((expanded) => !expanded);
                }}
                aria-expanded={recentChatsVisible}
                aria-controls="recent-chats-panel"
                aria-label={`${recentChatsVisible ? "Collapse" : "Expand"} Recent Chats`}
                title="Recent Chats"
              >
                <span className="recent-chats-icon" aria-hidden="true">
                  <Clock3 size={15} />
                </span>
                <span className="recent-chats-label">Recent Chats</span>
                <span className="recent-chats-chevron" aria-hidden="true">
                  {recentChatsVisible ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </span>
              </button>
              {recentChatsVisible && (
                <button
                  className="history-refresh-button"
                  type="button"
                  onClick={() => {
                    setHistoryError("");
                    setHistoryLoading(true);
                    refreshConversationList();
                  }}
                  disabled={historyLoading || conversationMutation !== ""}
                  aria-label="Refresh conversation history"
                  title="Refresh conversation history"
                >
                  <RefreshCw size={14} />
                </button>
              )}
            </div>

            <div
              className="recent-chats-panel"
              id="recent-chats-panel"
              hidden={!recentChatsVisible}
            >
              {historyLoading ? (
                <div className="conversation-history-state" role="status">
                  <span className="loading-spinner" />
                  Loading conversations…
                </div>
              ) : historyError ? (
                <div className="conversation-history-error" role="alert">
                  <span>{historyError}</span>
                  <button
                    type="button"
                    onClick={() => {
                      setHistoryError("");
                      setHistoryLoading(true);
                      refreshConversationList();
                    }}
                  >
                    Retry
                  </button>
                </div>
              ) : conversations.length === 0 ? (
                <p className="conversation-history-empty">
                  No conversations yet. Start a new chat to save your history.
                </p>
              ) : (
                <div className="conversation-list">
                  {conversations.map((conversation) => (
                <div
                  className={`conversation-row ${
                    activeConversationId === conversation.id
                      ? "conversation-row-active"
                      : ""
                  }`}
                  key={conversation.id}
                >
                  {renamingConversationId === conversation.id ? (
                    <form
                      className="conversation-rename-form"
                      onSubmit={(event) => {
                        event.preventDefault();
                        saveConversationTitle(conversation.id);
                      }}
                    >
                      <input
                        aria-label={`Rename ${conversation.title}`}
                        value={renameDraft}
                        maxLength={200}
                        autoFocus
                        onChange={(event) => setRenameDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            setRenamingConversationId(null);
                            setRenameDraft("");
                          }
                        }}
                      />
                      <button
                        type="submit"
                        disabled={
                          !renameDraft.trim() ||
                          conversationMutation === String(conversation.id) ||
                          busy ||
                          conversationLoading
                        }
                        aria-label="Save conversation name"
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setRenamingConversationId(null);
                          setRenameDraft("");
                        }}
                        aria-label="Cancel renaming"
                      >
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      <button
                        className="conversation-select-button"
                        type="button"
                        onClick={() => openConversation(conversation.id)}
                        disabled={
                          busy ||
                          conversationLoading ||
                          conversationMutation !== ""
                        }
                        aria-current={
                          activeConversationId === conversation.id
                            ? "page"
                            : undefined
                        }
                        title={conversation.title}
                      >
                        <span className="conversation-title">
                          {conversation.title}
                        </span>
                        <time dateTime={conversation.updated_at}>
                          {new Date(conversation.updated_at).toLocaleDateString()}
                        </time>
                      </button>
                      {confirmingDeleteId === conversation.id ? (
                        <div
                          className="conversation-delete-confirm"
                          role="group"
                          aria-label={`Confirm deleting ${conversation.title}`}
                        >
                          <span>Delete chat?</span>
                          <button
                            type="button"
                            onClick={() => removeConversation(conversation.id)}
                            disabled={
                              conversationMutation === String(conversation.id) ||
                              busy ||
                              conversationLoading
                            }
                          >
                            Delete
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmingDeleteId(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div className="conversation-actions">
                          <button
                            type="button"
                            onClick={() => {
                              setHistoryError("");
                              setConfirmingDeleteId(null);
                              setRenamingConversationId(conversation.id);
                              setRenameDraft(conversation.title);
                            }}
                            disabled={
                              conversationMutation !== "" ||
                              busy ||
                              conversationLoading
                            }
                            aria-label={`Rename ${conversation.title}`}
                            title="Rename conversation"
                          >
                            <Pencil size={13} />
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setHistoryError("");
                              setRenamingConversationId(null);
                              setConfirmingDeleteId(conversation.id);
                            }}
                            disabled={
                              conversationMutation !== "" ||
                              busy ||
                              conversationLoading
                            }
                            aria-label={`Delete ${conversation.title}`}
                            title="Delete conversation"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      )}
                    </>
                  )}
                  </div>
                  ))}
                </div>
              )}
            </div>
          </section>

        <div className="sidebar-footer">
          <div className="sidebar-bottom">
            <div className="local-status">
              <span className="status-dot" />
              <div>
                <strong>QuantaLM</strong>
                <small>
                  Local model · {quantaLoaded ? "Loaded" : "Loads on use"}
                </small>
              </div>
            </div>
          </div>
          <button
            type="button"
            className={`sidebar-profile-button ${
              activePage === "profile" ? "sidebar-profile-active" : ""
            }`}
            onClick={() => navigateTo("profile")}
            aria-label="Open My Profile"
            title="My Profile"
          >
            <span className="sidebar-profile-avatar" aria-hidden="true">
              {user.name.trim().charAt(0).toUpperCase()}
            </span>
            <span>My Profile</span>
          </button>
          <button
            type="button"
            className={`sidebar-settings-button ${
              activePage === "settings" ? "sidebar-settings-active" : ""
            }`}
            onClick={() => navigateTo("settings")}
            aria-label="Open Settings"
            title="Settings"
          >
            <SettingsIcon size={15} aria-hidden="true" />
            <span>Settings</span>
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          {activePage === "workspace" ? (
            <div className="topbar-identity">
            <div className="assistant-heading">
              <span className="assistant-avatar" aria-hidden="true">
                <selectedAgent.icon size={18} strokeWidth={1.8} />
              </span>
              <div className="assistant-heading-copy">
                <span className="eyebrow">QUANTAAI WORKSPACE</span>
                <h2>{selectedAgent.name}</h2>
              </div>
            </div>
            {collaborationMode && (
              <div className="selected-agent-summary" aria-label="Agents active for the next message">
                {activeParticipants.map((agent) => (
                  <span className="selected-agent-chip" key={agent.id} title={agent.name}>
                    <agent.icon size={12} aria-hidden="true" />
                    <span>{agent.name}</span>
                  </span>
                ))}
                <span className="collaboration-label">COLLABORATION</span>
              </div>
            )}
              </div>
          ) : (
              <div className="topbar-page-heading">
                <div>
                  <span className="eyebrow">QUANTAAI ACCOUNT</span>
                  <h2>{pageTitles[activePage] || "My Profile"}</h2>
                </div>
              </div>
          )}

          <div className="topbar-controls">
              {activePage === "workspace" && (
                <>
              <button
                className="topbar-new-chat"
              type="button"
              onClick={startNewChat}
              disabled={busy || historyLoading || conversationLoading || conversationMutation !== ""}
            >
              <Plus size={15} aria-hidden="true" />
              <span>New chat</span>
            </button>
            <label className="mode-selector">
              <span>Mode</span>
              <select
                value={chatMode}
                onChange={(event) => setChatMode(event.target.value)}
                disabled={busy}
                aria-label="Chat mode"
              >
                {modes.map((mode) => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label}
                  </option>
                ))}
              </select>
            </label>
            {activeAgent === "Study" && !collaborationMode && (
              <label className="mode-selector">
                <span>Tutor</span>
                <select
                  value={tutorMode}
                  onChange={(event) => setTutorMode(event.target.value)}
                  disabled={busy}
                  aria-label="Study tutor mode"
                >
                  <option value="direct">Direct</option>
                  <option value="socratic">Socratic</option>
                  <option value="deep_learning">Deep Learning</option>
                </select>
              </label>
            )}
            <div className="model-badge" title={backendMessage}>
              <span
                className={`connection-dot ${
                  backendStatus === "Connected" ? "connected" : ""
                }`}
              />
              {backendStatus}
            </div>
              </>
            )}
            <div className="account-menu" ref={accountMenuRef}>
              <button
                type="button"
                className="account-avatar-button"
                onClick={() => {
                  setAccountMenuOpen((open) => !open);
                  setAccountMenuError("");
                }}
                aria-label="Open account menu"
                aria-haspopup="menu"
                aria-expanded={accountMenuOpen}
                title={user.name}
              >
                {user.name.trim().charAt(0).toUpperCase()}
              </button>
              {accountMenuOpen && (
                <div className="account-dropdown" role="menu" aria-label="Account menu">
                  <div className="account-dropdown-user">
                    <strong>{user.name}</strong>
                    <span>{user.email}</span>
                  </div>
                  <button type="button" role="menuitem" onClick={() => navigateTo("profile")}>
                    My Profile
                  </button>
                  <button type="button" role="menuitem" onClick={() => navigateTo("settings")}>
                    Settings
                  </button>
                  {accountMenuError && (
                    <p className="account-menu-error" role="alert">{accountMenuError}</p>
                  )}
                  <button
                    type="button"
                    role="menuitem"
                    className="account-signout"
                    onClick={() => {
                      handleLogout().catch((error) => {
                        setAccountMenuError(
                          error.message || "Unable to sign out. Please try again.",
                        );
                      });
                    }}
                  >
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        {activePage === "settings" ? (
          <Settings
            theme={theme}
            onThemeChange={setTheme}
            chatMode={chatMode}
            onChatModeChange={setChatMode}
            modes={modes}
            backendStatus={backendStatus}
            teacherAvailable={teacherAvailable}
            quantaLoaded={quantaLoaded}
            version={frontendPackage.version}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "profile" ? (
          <Profile
            user={user}
            onUpdate={handleProfileUpdate}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "planner" ? (
          <Planner
            tasks={tasks}
            loading={plannerLoading}
            error={plannerError}
            mutationId={taskMutation}
            reminderTimes={plannerReminderTimes}
            reminderError={plannerReminderError}
            browserNotificationsEnabled={browserNotificationsEnabled}
            onEnableBrowserNotifications={enableBrowserNotifications}
            onReminderTimeChange={handleReminderTimeChange}
            onCreate={handleCreateTask}
            onUpdate={handleUpdateTask}
            onDelete={handleDeleteTask}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "quiz" ? (
          <Quiz
            mode={chatMode}
            onAuthenticationExpired={handleAuthenticationExpired}
            onProgressUpdated={refreshLearningProgress}
            launchRequest={quizLaunchRequest}
            onLaunchHandled={clearQuizLaunchRequest}
            onReviewFlashcards={(deckId) => {
              setFlashcardDeckToReview(deckId);
              navigateTo("flashcards");
            }}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "flashcards" ? (
          <Flashcards
            mode={chatMode}
            onAuthenticationExpired={handleAuthenticationExpired}
            onProgressUpdated={refreshLearningProgress}
            initialDeckId={flashcardDeckToReview}
            onAttemptQuiz={launchQuizFromDeck}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "documents" ? (
          <DocumentLibrary
            onAuthenticationExpired={handleAuthenticationExpired}
            onDocumentsChanged={handleDocumentsChanged}
            onBack={() => navigateTo("workspace")}
          />
        ) : activePage === "progress" ? (
          <Progress
            onAuthenticationExpired={handleAuthenticationExpired}
            onBack={() => navigateTo("workspace")}
          />
        ) : (
          <>
        <section className="chat-area">
          {conversationLoading ? (
            <div className="conversation-loading-state" role="status">
              <span className="loading-spinner" />
              Loading conversation…
            </div>
          ) : messages.length === 0 ? (
            <div className="welcome">
              <img
                className="welcome-orb"
                src="/quantaai-icon.png"
                alt="QuantaAI logo"
              />
              <span className="eyebrow">
                {activeAgent === surfaceAssistant.id
                  ? "SURFACE AI · GENERAL ASSISTANT"
                  : `${selectedAgent.name.toUpperCase()} · SPECIALIST ASSISTANT`}
              </span>
              <h2>
                {activeAgent === surfaceAssistant.id ? (
                  <>Welcome to <span>Surface AI.</span></>
                ) : (
                  <>Let&apos;s make <span>progress together.</span></>
                )}
              </h2>
              <p>
                {activeAgent === surfaceAssistant.id
                  ? "Ask anything, explore an idea, or tell me what you are working on."
                  : selectedAgent.desc}
              </p>

              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button
                    type="button"
                    className="suggestion-card"
                    key={item.title}
                    onClick={() => sendMessage(item.prompt)}
                  >
                    <span className="suggestion-icon">
                      <item.icon size={17} aria-hidden="true" />
                    </span>
                    <strong>{item.title}</strong>
                    <span>{item.text}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((item) => (
                <article
                  className={`message ${
                    item.role === "user"
                      ? "user-message"
                      : item.role === "error"
                        ? "error-message"
                        : "assistant-message"
                  }`}
                  key={item.id}
                >
                  <div className="message-avatar">
                    {item.role === "user"
                      ? "Y"
                      : (() => {
                          const MessageIcon = item.icon || selectedAgent.icon;
                          return <MessageIcon size={16} aria-hidden="true" />;
                        })()}
                  </div>
                  <div>
                    <div className="message-meta">
                      <div className="message-attribution">
                        <strong>
                          {item.role === "user"
                            ? "You"
                            : item.role === "error"
                              ? `${item.agent || "Agent"} · Didn't respond`
                              : item.agent}
                        </strong>
                      </div>
                      <time dateTime={item.timestamp}>
                        {formatMessageTime(item.timestamp)}
                      </time>
                    </div>
                    {item.documentAttachments.length > 0 && (
                      <div
                        className="conversation-pdf-attachments"
                        aria-label="PDF attachments"
                      >
                        {item.documentAttachments.map((attachment) => (
                          <span
                            className="conversation-pdf-attachment"
                            key={attachment.id}
                          >
                            <FileText size={14} aria-hidden="true" />
                            <span>{attachment.filename}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="message-content">
                      {item.role === "assistant"
                        ? <MarkdownReply content={item.text} />
                        : <p>{item.text}</p>}
                    </div>
                    {item.role === "assistant" &&
                      item.agent === "Planner Agent" &&
                      /(?:Added \d+ tasks? to Planner & To-do|already in Planner & To-do)/i.test(
                        item.text,
                      ) && (
                        <button
                          type="button"
                          className="planner-open-from-chat"
                          onClick={() => navigateTo("planner")}
                        >
                          Open Planner &amp; To-do
                        </button>
                      )}
                    {item.documentSource && item.role === "assistant" && (
                      <p className="document-source-note" role="note">
                        Source: {item.documentSource === "uploaded_pdf"
                          ? "Uploaded PDF"
                          : item.documentSource === "pdf_ai_knowledge"
                            ? "PDF + AI Knowledge"
                            : "AI Knowledge"}
                        {item.documentPages.length > 0 && (
                          <> · PDF pages {item.documentPages.join(", ")}</>
                        )}
                      </p>
                    )}
                    {item.reliabilityWarning && item.role === "assistant" && (
                      <p className="quanta-reliability-warning" role="note">
                        This response may be inaccurate. QuantaLM is still learning.
                      </p>
                    )}
                    {item.searchMessage && item.role === "assistant" && (
                      <p className="research-search-note" role="note">
                        {item.searchMessage}
                      </p>
                    )}
                    {item.sources.length > 0 && item.role === "assistant" && (
                      <section className="research-sources" aria-label="Web research sources">
                        <h3>Sources · {item.searchAgent || item.agent}</h3>
                        <ol>
                          {item.sources.map((source) => (
                            <li key={source.id || source.url}>
                              <a href={source.url} target="_blank" rel="noreferrer">
                                <span>[{source.id}]</span> {source.title}
                              </a>
                            </li>
                          ))}
                        </ol>
                      </section>
                    )}
                    {item.agent === "Study Agent" &&
                      item.role === "assistant" &&
                      (item.videoMessage || item.videos.length > 0) && (
                        <section className="study-videos" aria-label="Recommended YouTube videos">
                          <h3>Recommended YouTube Videos</h3>
                          {item.videoMessage && item.videos.length === 0 && (
                            <p className="study-video-message">{item.videoMessage}</p>
                          )}
                          {item.videos.length > 0 && (
                            <div className="study-video-grid">
                              {item.videos.map((video) => (
                                <a
                                  className="study-video-card"
                                  href={video.url}
                                  key={video.id}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <img src={video.thumbnail} alt="" loading="lazy" />
                                  <span className="study-video-details">
                                    <strong>{video.title}</strong>
                                    <small>{video.channel}</small>
                                  </span>
                                </a>
                              ))}
                            </div>
                          )}
                        </section>
                      )}
                  </div>
                </article>
              ))}
              {busy && (
                <ProcessingIndicator
                  agents={respondingAgents}
                  searching={searchingAgents.length > 0}
                />
              )}
            </div>
          )}
        </section>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage();
          }}
        >
          {activeAgent === "Study" && !collaborationMode && (
            <div className="composer-document-controls">
                  <input
                    ref={studyDocumentInputRef}
                    className="study-document-file-input"
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={handleStudyPdfUpload}
                    aria-label="Choose a PDF to upload"
                    disabled={studyDocumentUploading || busy}
                  />

                  {selectedStudyDocument && (
                    <div className="composer-document-attachment">
                      <span className="composer-document-type">PDF</span>
                      <span className="composer-document-name">
                        <strong title={selectedStudyDocument.filename}>
                          {selectedStudyDocument.filename}
                        </strong>
                        <small>Indexed and attached</small>
                      </span>
                      <button
                        type="button"
                        className="composer-document-remove"
                        onClick={() => setSelectedStudyDocumentId(null)}
                        disabled={busy}
                        aria-label={`Remove ${selectedStudyDocument.filename} attachment`}
                        title="Remove attachment"
                      >
                        ×
                      </button>
                    </div>
                  )}

                  {studyDocumentUploading && (
                    <div className="study-document-progress" role="status">
                      <span>
                        {studyDocumentProgress >= 100
                          ? "Processing and indexing PDF…"
                          : studyDocumentProgress > 0
                            ? `Uploading PDF… ${studyDocumentProgress}%`
                            : "Preparing PDF upload…"}
                      </span>
                      <progress
                        max="100"
                        value={studyDocumentProgress || undefined}
                        aria-label="PDF upload progress"
                      />
                    </div>
                  )}

                  {studyDocumentError && (
                    <p className="study-document-error" role="alert">
                      {studyDocumentError}
                    </p>
                  )}

                  {studyDocumentMenuOpen && (
                    <div className="study-document-menu" role="menu">
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => {
                          setStudyDocumentMenuOpen(false);
                          studyDocumentInputRef.current?.click();
                        }}
                        disabled={studyDocumentUploading || busy}
                      >
                        Upload PDF
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => {
                          setStudyDocumentMenuOpen(false);
                          setStudyDocumentPickerOpen(true);
                        }}
                        disabled={studyDocumentUploading || busy}
                      >
                        Select existing document
                      </button>
                    </div>
                  )}

                  {studyDocumentPickerOpen && (
                    <div
                      className="existing-document-picker"
                      role="dialog"
                      aria-label="Select an existing document"
                    >
                      <div className="existing-document-picker-heading">
                        <strong>Select existing document</strong>
                        <button
                          type="button"
                          className="existing-document-picker-close"
                          onClick={() => setStudyDocumentPickerOpen(false)}
                          aria-label="Close document picker"
                        >
                          ×
                        </button>
                      </div>
                      {studyDocumentsLoading ? (
                        <span className="study-document-list-status existing-document-list-status" role="status">
                          Loading your PDFs…
                        </span>
                      ) : studyDocuments.length === 0 ? (
                        <span className="study-document-list-status existing-document-list-status">
                          No PDFs uploaded yet.
                        </span>
                      ) : (
                        <div className="study-document-list existing-document-list" aria-label="Available PDFs">
                          {studyDocuments.map((document) => (
                            <div className="study-document-row existing-document-row" key={document.id}>
                              <button
                                type="button"
                                className="existing-document-select"
                                onClick={() => {
                                  setSelectedStudyDocumentId(document.id);
                                  setStudyDocumentPickerOpen(false);
                                }}
                                disabled={
                                  document.status !== "indexed" ||
                                  busy ||
                                  studyDocumentUploading
                                }
                              >
                                <strong title={document.filename}>
                                  {document.filename}
                                </strong>
                                <span
                                  className={`study-document-status ${document.status}`}
                                >
                                  {document.status === "indexed"
                                    ? "Indexed"
                                    : document.status === "processing"
                                      ? "Processing"
                                      : "Could not index"}
                                </span>
                                {document.status_message && (
                                  <small>{document.status_message}</small>
                                )}
                              </button>
                              {studyDocumentDeleteConfirmationId === document.id ? (
                                <div
                                  className="existing-document-delete-confirm"
                                  role="group"
                                  aria-label={`Confirm deleting ${document.filename}`}
                                >
                                  <span>Delete this PDF?</span>
                                  <button
                                    type="button"
                                    onClick={() => removeStudyDocument(document)}
                                    disabled={
                                      busy ||
                                      studyDocumentUploading ||
                                      studyDocumentMutationId === document.id
                                    }
                                  >
                                    {studyDocumentMutationId === document.id
                                      ? "Deleting…"
                                      : "Confirm"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setStudyDocumentDeleteConfirmationId(null)
                                    }
                                  >
                                    Cancel
                                  </button>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  className="study-document-remove existing-document-remove"
                                  onClick={() =>
                                    setStudyDocumentDeleteConfirmationId(document.id)
                                  }
                                  disabled={
                                    busy ||
                                    studyDocumentUploading ||
                                    studyDocumentMutationId === document.id
                                  }
                                  aria-label={`Delete ${document.filename}`}
                                >
                                  Delete
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
            </div>
          )}

          {activeAgent === surfaceAssistant.id && !collaborationMode && (
            <div className="composer-document-controls surface-document-controls">
              <input
                ref={studyDocumentInputRef}
                className="study-document-file-input"
                type="file"
                accept="application/pdf,.pdf"
                multiple
                onChange={handleSurfacePdfUpload}
                aria-label="Choose PDFs for Surface AI"
                disabled={surfaceDocumentUploading || busy}
              />
              {surfaceAttachments.length > 0 && (
                <div className="surface-document-attachments">
                  {surfaceAttachments.map((attachment) => (
                    <div className="composer-document-attachment" key={attachment.id}>
                      <span className="composer-document-type">PDF</span>
                      <span className="composer-document-name">
                        <strong title={attachment.filename}>
                          {attachment.filename}
                        </strong>
                        <small>Ready to attach to this conversation</small>
                      </span>
                      <button
                        type="button"
                        className="composer-document-remove"
                        onClick={() =>
                          setSurfaceAttachments((previous) =>
                            previous.filter((item) => item.id !== attachment.id),
                          )
                        }
                        disabled={surfaceDocumentUploading || busy}
                        aria-label={`Remove ${attachment.filename} attachment`}
                        title="Remove attachment"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {surfaceDocumentUploading && (
                <div className="study-document-progress" role="status">
                  <span>
                    {surfaceDocumentProgress >= 100
                      ? `Processing and indexing ${surfaceUploadingFilename}…`
                      : surfaceDocumentProgress > 0
                        ? `Uploading ${surfaceUploadingFilename}… ${surfaceDocumentProgress}%`
                        : "Preparing PDF upload…"}
                  </span>
                  <progress
                    max="100"
                    value={surfaceDocumentProgress || undefined}
                    aria-label="Surface PDF upload progress"
                  />
                </div>
              )}
              {surfaceDocumentError && (
                <p className="study-document-error" role="alert">
                  {surfaceDocumentError}
                </p>
              )}
              <button
                type="button"
                className="surface-existing-document-button"
                onClick={() => setStudyDocumentPickerOpen((open) => !open)}
                disabled={surfaceDocumentUploading || busy}
                aria-expanded={studyDocumentPickerOpen}
              >
                Select an uploaded PDF
              </button>
              {studyDocumentPickerOpen && (
                <div
                  className="existing-document-picker surface-document-picker"
                  role="dialog"
                  aria-label="Select an uploaded PDF for Surface AI"
                >
                  <div className="existing-document-picker-heading">
                    <strong>Your uploaded PDFs</strong>
                    <button
                      type="button"
                      className="existing-document-picker-close"
                      onClick={() => setStudyDocumentPickerOpen(false)}
                      aria-label="Close document picker"
                    >
                      ×
                    </button>
                  </div>
                  {studyDocumentsLoading ? (
                    <span className="existing-document-list-status" role="status">Loading PDFs…</span>
                  ) : studyDocuments.filter((item) => item.status === "indexed").length === 0 ? (
                    <span className="existing-document-list-status">No indexed PDFs available.</span>
                  ) : (
                    <div className="study-document-list existing-document-list" aria-label="Available PDFs">
                      {studyDocuments
                        .filter((item) => item.status === "indexed")
                        .map((document) => {
                          const isAttached = surfaceAttachments.some(
                            (attachment) => attachment.id === document.id,
                          );
                          return (
                            <div className="study-document-row existing-document-row" key={document.id}>
                              <button
                                type="button"
                                className="existing-document-select surface-document-select"
                                disabled={
                                  isAttached ||
                                  surfaceAttachments.length >= 5 ||
                                  surfaceDocumentUploading ||
                                  busy
                                }
                                onClick={() => {
                                  setSurfaceAttachments((previous) =>
                                    previous.some((item) => item.id === document.id)
                                      ? previous
                                      : [...previous, {
                                          id: document.id,
                                          filename: document.filename,
                                        }],
                                  );
                                  setStudyDocumentPickerOpen(false);
                                  setSurfaceDocumentError("");
                                }}
                              >
                                <FileText size={15} aria-hidden="true" />
                                <strong title={document.filename}>{document.filename}</strong>
                                <small>{isAttached ? "Attached" : "Indexed"}</small>
                              </button>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="composer-input-row">
            {activeAgent === "Study" && !collaborationMode && (
              <button
                type="button"
                className="composer-add-button"
                onClick={() => {
                  setStudyDocumentPickerOpen(false);
                  setStudyDocumentMenuOpen((open) => !open);
                }}
                disabled={studyDocumentUploading || busy}
                aria-label="Add a study document"
                aria-expanded={studyDocumentMenuOpen || studyDocumentPickerOpen}
                title="Add a study document"
              >
                +
              </button>
            )}
            {activeAgent === surfaceAssistant.id && !collaborationMode && (
              <button
                type="button"
                className="composer-add-button surface-composer-add-button"
                onClick={() => studyDocumentInputRef.current?.click()}
                disabled={surfaceDocumentUploading || busy}
                aria-label="Attach a PDF to Surface AI"
                title="Attach a PDF"
              >
                <Paperclip size={17} aria-hidden="true" />
              </button>
            )}
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={
                collaborationMode
                  ? selectedAgents.length > 0
                    ? `Message ${selectedAgents.length} selected agent${selectedAgents.length === 1 ? "" : "s"}…`
                    : "Choose agents in the sidebar to collaborate…"
                  : `Message ${selectedAgent.name}…`
              }
              rows={1}
              aria-label="Message"
              disabled={
                busy ||
                surfaceDocumentUploading ||
                conversationLoading ||
                historyLoading ||
                (collaborationMode && selectedAgents.length === 0)
              }
            />

            <button
              type="submit"
              disabled={
                !message.trim() ||
                busy ||
                surfaceDocumentUploading ||
                conversationLoading ||
                historyLoading
              }
              aria-label="Send message"
            >
              <ArrowUp size={18} aria-hidden="true" />
            </button>
          </div>

        </form>

        <footer className="main-footer">
          {notice} · Built with your own AI architecture
        </footer>
          </>
        )}
      </main>

      <aside className="right-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">YOUR JOURNEY</span>
            <h2>Learning dashboard</h2>
          </div>
        </div>

        <section className="dashboard-card">
          <span className="card-label">TODAY'S FOCUS</span>
          <div className="focus-illustration" aria-hidden="true">
            <Sparkles size={19} />
          </div>
          <h3>
            {learningProgress?.recommendations?.[0]?.topic
              ? `Review ${learningProgress.recommendations[0].topic}`
              : "No focus set yet"}
          </h3>
          <p>
            {progressError ||
              learningProgress?.recommendations?.[0]?.message ||
              "Complete a quiz to get a suggested review focus."}
          </p>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="Average quiz score"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={learningProgress?.average_score || 0}
          >
            <span
              style={{
                width: `${learningProgress?.average_score || 0}%`,
              }}
            />
          </div>
          <div className="progress-caption">
            <span>Average quiz score</span>
            <strong>
              {learningProgress?.average_score === null ||
              learningProgress?.average_score === undefined
                ? "Not available"
                : `${learningProgress.average_score}%`}
            </strong>
          </div>
        </section>

        <section className="dashboard-section">
          <div className="section-title">
            <h3>Upcoming tasks</h3>
            <button
              type="button"
              onClick={() => setShowTasks(!showTasks)}
            >
              {showTasks ? "Close" : "+ Add"}
            </button>
          </div>

          {showTasks && (
            <form className="task-form" onSubmit={addTask}>
              <input
                value={taskTitle}
                onChange={(event) => setTaskTitle(event.target.value)}
                placeholder="Task title"
                maxLength={200}
                required
              />
              <input
                type="date"
                value={taskDate}
                onChange={(event) => setTaskDate(event.target.value)}
                required
              />
              <button
                type="submit"
                disabled={!taskTitle.trim() || !taskDate || taskMutation !== ""}
              >
                Save task
              </button>
            </form>
          )}

          {plannerError && <p className="learning-error" role="alert">{plannerError}</p>}

          {plannerLoading ? (
            <p className="learning-state" role="status">Loading tasks…</p>
          ) : upcomingTasks.length === 0 ? (
            <div className="empty-tasks">
              <Clock3 size={16} aria-hidden="true" />
              <p>No deadlines yet.</p>
              <small>Add a task to organize your learning.</small>
            </div>
          ) : (
            <div className="task-list">
              {upcomingTasks.map((task) => (
                <label
                  className={`task-item ${
                    task.completed ? "task-done" : ""
                  }`}
                  key={task.id}
                >
                  <input
                    type="checkbox"
                    checked={task.completed}
                    onChange={(event) =>
                      handleUpdateTask(task.id, {
                        completed: event.target.checked,
                      })
                    }
                    disabled={taskMutation === String(task.id)}
                  />
                  <span>
                    <strong>{task.title}</strong>
                    {task.description && <small>{task.description}</small>}
                    <small>
                      {new Date(`${task.due_date}T00:00:00`).toLocaleDateString()}
                    </small>
                  </span>
                </label>
              ))}
            </div>
          )}
          <button
            type="button"
            className="upcoming-open-planner"
            onClick={() => navigateTo("planner")}
          >
            Open Planner &amp; To-do
          </button>
        </section>

        <section className="dashboard-section quick-info">
          <h3>System status</h3>
          <div>
            <span>Frontend</span>
            <strong className="status-good">Ready</strong>
          </div>
          <div>
            <span>FastAPI</span>
            <strong>{backendStatus}</strong>
          </div>
          <div>
            <span>Online AI</span>
            <strong>{teacherAvailable ? "Available" : "Not configured"}</strong>
          </div>
          <div>
            <span>QuantaLM</span>
            <strong>{quantaLoaded ? "Loaded" : "Loads on use"}</strong>
          </div>
          <div>
            <span>YouTube resources</span>
            <strong>{youtubeAvailable ? "Configured" : "Needs API key"}</strong>
          </div>
        </section>

        <div className="privacy-note">
          <Sparkles size={15} aria-hidden="true" />
          <p>
            <strong>Local-first by design</strong>
            <br />
            OFFLINE mode uses your local QuantaLM model.
          </p>
        </div>

      </aside>
    </div>
  );
}

export default App;