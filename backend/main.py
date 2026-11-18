import os
import json
import html
import logging
import re
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

from backend.auth import (
    User,
    get_db,
    require_csrf,
    require_user,
    router as auth_router,
)
from backend.collaboration import router as collaboration_router
from backend.conversations import (
    conversation_document_attachments,
    router as conversations_router,
)

from backend.agents.study_agent import StudyAgent
from backend.agents.profiles import (
    AGENT_PROMPTS,
    STUDY_TUTOR_MODE_PROMPTS,
    build_offline_prompt,
)
from backend.documents import (
    StudyDocumentContext,
    get_study_document_retrieval,
    resolve_document_answer,
    router as study_documents_router,
)
from backend.learning import router as learning_router
from backend.planner import router as planner_router
from backend.services.quanta_service import quanta_service
from backend.services.provider_errors import ProviderError, ProviderUnavailableError
from backend.services.provider_manager import provider_manager, select_provider
from backend.services.web_search_service import (
    build_research_context,
    should_search_web,
    web_search_service,
)
from backend.services.teacher_service import (
    teacher_service,
)
from backend.services.youtube_service import youtube_service


logger = logging.getLogger(__name__)


app = FastAPI(
    title="QuantaAI API",
    description="Backend for QuantaLM and AI Agents",
    version="1.0.0"
)

study_agent = StudyAgent()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "QUANTAAI_FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)

app.include_router(auth_router)
app.include_router(collaboration_router)
app.include_router(conversations_router)
app.include_router(study_documents_router)
app.include_router(learning_router)
app.include_router(planner_router)


AGENTS = [
    "Surface",
    "Study",
    "Coding",
    "Career",
    "Guide",
    "Planner",
    "Exam",
    "Research"
]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str
    agent: Literal[
        "Surface",
        "Study",
        "Coding",
        "Career",
        "Guide",
        "Planner",
        "Exam",
        "Research",
    ] = "Study"
    mode: Literal["auto", "offline", "teacher"] = "auto"
    tutor_mode: Literal["direct", "socratic", "deep_learning"] = "direct"
    conversation_history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=12,
    )
    document_id: int | None = Field(default=None, gt=0)
    conversation_id: int | None = Field(default=None, gt=0)

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        return message


class ScheduleTask(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    due_date: date

    @field_validator("title", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Task text cannot be empty.")
        return stripped


class ScheduleGeneration(BaseModel):
    tasks: list[dict[str, object]] = Field(max_length=60)
    clarification: str | None = Field(default=None, max_length=1000)


class ChatResponse(BaseModel):
    reply: str
    agent: str
    model: str
    provider: str
    status: str
    experimental: bool = False
    reliability_warning: bool = False
    videos: list[dict[str, str]] = Field(default_factory=list)
    video_message: str | None = None
    search_performed: bool = False
    search_agent: str | None = None
    search_status: str | None = None
    search_message: str | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)
    document_source: Literal[
        "uploaded_pdf",
        "pdf_ai_knowledge",
        "ai_knowledge",
    ] | None = None
    document_pages: list[int] = Field(default_factory=list)
    schedule_tasks: list[ScheduleTask] = Field(default_factory=list)
    schedule_tasks_error: str | None = None


PLANNER_TIMEZONE = ZoneInfo("Asia/Kolkata")
MAX_SCHEDULE_WINDOW_DAYS = 14
MAX_SCHEDULE_TASKS = 60


def _planner_system_prompt(current_date: date) -> str:
    return (
        f"{AGENT_PROMPTS['Planner']}\n\n"
        f"Backend current date: {current_date.isoformat()} (Asia/Kolkata). "
        "Use this explicit date, not dates inferred from memory or training data. "
        "Create a practical schedule beginning today or later. When adapting an "
        "old study plan, recognize completed phases and historical milestones, "
        "keep the user's goals and unfinished syllabus, discard past due dates, "
        "and redistribute remaining work across the available future time. Do not "
        "invent a confirmed examination date; ask the user if a critical deadline "
        "is uncertain. Give the user a concise readable overview, but do not claim "
        "that tasks have been saved or added. Structured task data is generated "
        "separately."
    )


def _is_schedule_request(message: str) -> bool:
    schedule_action = re.search(
        r"\b(?:create|make|build|generate|organize|plan|schedule|add|"
        r"include|assign|put|set\s+up)\b",
        message,
        re.IGNORECASE,
    )
    schedule_subject = re.search(
        r"\b(?:schedule|timetable|routine|planner|task(?:s)?|activities|"
        r"study|preparation|exam(?:ination)?|semester|revision|revise|"
        r"PYQs?|practice|mock\s+tests?|GATE|JEE|UPSC|NEET|CAT|"
        r"\d{1,2}\s*[- ]?\s*days?|weeks?)\b",
        message,
        re.IGNORECASE,
    )
    add_to_planner = re.search(
        r"\badd\b.{0,100}\b(?:to|into)\s+(?:my\s+)?planner\b",
        message,
        re.IGNORECASE,
    )
    date_specific_activity = re.search(
        r"\b(?:tomorrow|today|this\s+(?:week|weekend)|next\s+\d+\s+days?|"
        r"\d{1,2}[- ]day|daily|weekly|weekend)\b",
        message,
        re.IGNORECASE,
    )
    return bool(
        add_to_planner
        or (
            schedule_action
            and schedule_subject
            and (
                schedule_subject.group(0).casefold() not in {"planner", "task", "tasks"}
                or date_specific_activity
            )
        )
    )


def _current_planner_date() -> date:
    return datetime.now(PLANNER_TIMEZONE).date()


def _schedule_window_days(message: str) -> int:
    match = re.search(
        r"\b(?:(?:for|next|upcoming|within|a|an)\s+)?"
        r"(\d{1,2})[-\s]*(day|week)s?\b",
        message,
        re.IGNORECASE,
    )
    if match:
        days = int(match.group(1))
        if match.group(2).casefold() == "week":
            days *= 7
        return min(max(days, 1), MAX_SCHEDULE_WINDOW_DAYS)
    if re.search(r"\b(?:one|a)\s+week\b|\bweekly\b", message, re.IGNORECASE):
        return 7
    return 7


def _needs_exam_deadline_clarification(
    message: str,
    history: list[dict[str, str]],
) -> bool:
    user_context = "\n".join(
        [message]
        + [
            item["content"]
            for item in history
            if item.get("role") == "user"
        ]
    )
    named_exam = re.search(
        r"\b(?:GATE|JEE|UPSC|NEET|CAT)\s*20\d{2}\b|"
        r"\b20\d{2}\s+(?:GATE|JEE|UPSC|NEET|CAT)\b",
        user_context,
        re.IGNORECASE,
    )
    if named_exam is None:
        return False
    needs_full_horizon = re.search(
        r"\b(?:until|through|till|by)\s+(?:the\s+)?(?:exam|examination)|"
        r"\b(?:exam|examination)\s+date\b|"
        r"\b(?:complete|entire|full|long[- ]term)\s+"
        r"(?:exam\s+)?(?:preparation|study)\s+(?:plan|schedule|timetable)\b",
        message,
        re.IGNORECASE,
    )
    if needs_full_horizon is None:
        return False
    explicit_date = (
        r"(?:\b20\d{2}-\d{2}-\d{2}\b|"
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?"
        r"\s+20\d{2}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|"
        r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+20\d{2}\b)"
    )
    deadline_label = (
        r"(?:\b(?:exam|examination|test)\s+date\b|"
        r"\bdate\s+of\s+(?:the\s+)?(?:exam|examination|test)\b|"
        r"\b(?:exam|examination|test)\s+(?:is\s+)?scheduled\s+(?:for|on)\b)"
    )
    date_context = rf"(?:{deadline_label}[^\n.!?]{{0,80}}{explicit_date}|"
    date_context += rf"{explicit_date}[^\n.!?]{{0,80}}{deadline_label})"
    return re.search(date_context, user_context, re.IGNORECASE) is None


def _parse_schedule_generation(raw_response: str) -> ScheduleGeneration:
    candidate = raw_response.strip()
    fenced_json = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        re.IGNORECASE | re.DOTALL,
    )
    if fenced_json:
        candidate = fenced_json.group(1).strip()
    candidate = re.sub(r"\\([*_`~])", r"\1", candidate)

    def decode_json(text: str) -> object:
        start = text.find("{")
        if start < 0:
            raise json.JSONDecodeError("JSON object not found", text, 0)
        parsed, end = json.JSONDecoder().raw_decode(text, start)
        if text[end:].strip():
            raise json.JSONDecodeError("Unexpected content after JSON object", text, end)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed

    try:
        try:
            parsed = decode_json(candidate)
        except json.JSONDecodeError:
            decoded_candidate = html.unescape(candidate)
            if decoded_candidate == candidate:
                raise
            parsed = decode_json(decoded_candidate)

        def decode_entities(value: object) -> object:
            if isinstance(value, str):
                return html.unescape(value)
            if isinstance(value, list):
                return [decode_entities(item) for item in value]
            if isinstance(value, dict):
                return {
                    key: decode_entities(item)
                    for key, item in value.items()
                }
            return value

        parsed = decode_entities(parsed)
        return ScheduleGeneration.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise ValueError(
            "The schedule task generator returned invalid structured data."
        ) from error


def _render_schedule_tasks(tasks: list[ScheduleTask]) -> str:
    lines = [
        "Here is your timetable. Each item below is the exact task and due date "
        "that will be added to Planner & To-do:"
    ]
    for task in sorted(tasks, key=lambda item: item.due_date):
        lines.append(
            f"- **{task.due_date.isoformat()} — {task.title}**\n"
            f"  {task.description}"
        )
    return "\n\n".join(lines)


def _planner_task_generation_prompt(
    *,
    current_date: date,
    window_days: int,
) -> str:
    window_end = current_date + timedelta(days=window_days - 1)
    return (
        "Generate only JSON matching this shape: "
        '{"tasks":[{"title":"...","description":"...","due_date":"YYYY-MM-DD"}],'
        '"clarification":null}. Do not use Markdown fences, comments, or extra keys. '
        f"Today's date is {current_date.isoformat()} in Asia/Kolkata. "
        f"Create tasks only from {current_date.isoformat()} through "
        f"{window_end.isoformat()} inclusive (at most {window_days} days). "
        f"Never emit more than {MAX_SCHEDULE_TASKS} tasks. If the user gave an old "
        "or comprehensive plan, separate completed/historical work from unfinished "
        "goals, ignore all past dates, and reschedule only unfinished topics within "
        "this future window. Respect the user's subjects, routine, and time limits. "
        "For exam preparation, when the user requests daily/weekly study habits, "
        "create actionable dated occurrences suited to the requested window (for "
        "example distinct daily learning/practice/review work and one or two "
        "weekend revision/test tasks when feasible); do not put every milestone on "
        "one date. Use the actual Saturday/Sunday dates for weekend tasks and "
        "create a separate task occurrence for every requested daily activity. "
        "The generated dates are canonical: never shift or reformat them in task "
        "data. For an exam-specific long-range plan, create only the first "
        "window of concrete tasks and milestones; the user can request the next "
        "portion later. A confirmed official exam date is not needed for a "
        "bounded study window; do not invent one, and ask for it only when the "
        "user explicitly asks to schedule through that exam or against its "
        "official deadline. If the request lacks essential information, return no "
        "tasks and set clarification. Every task must have a concise title, a "
        "specific description, and a valid ISO date in the allowed window. "
        "Do not repeat identical title/date tasks. Priority is not supported by the "
        "current planner, so do not add it."
    )


def _generate_schedule_tasks(
    *,
    request: ChatRequest,
    history: list[dict[str, str]],
    provider: str,
    current_date: date,
) -> tuple[list[ScheduleTask], str | None]:
    if _needs_exam_deadline_clarification(request.message, history):
        return (
            [],
            "What is the confirmed examination date? I won't assume an official "
            "date for this exam.",
        )

    window_days = _schedule_window_days(request.message)
    task_prompt = _planner_task_generation_prompt(
        current_date=current_date,
        window_days=window_days,
    )
    task_request = (
        f"User's timetable request and any supplied plan:\n"
        f"{request.message[:16000]}\n\n"
        f"Generate the first {window_days}-day structured task portion now."
    )
    try:
        if provider == "quanta":
            raw_response = quanta_service.generate(
                build_offline_prompt(
                    "Planner",
                    task_request,
                    conversation_history=history,
                    system_prompt=task_prompt,
                ),
                max_new_tokens=2600,
            )
        else:
            generated = provider_manager.generate(
                task_request,
                "Planner",
                history,
                system_prompt=task_prompt,
                max_tokens=2600,
                temperature=0.1,
            )
            raw_response = generated["reply"]

        schedule = _parse_schedule_generation(raw_response)
    except (ProviderError, ValueError, TypeError, RuntimeError) as error:
        logger.warning(
            "Planner structured schedule generation failed error_type=%s",
            type(error).__name__,
        )
        return [], (
            "The timetable could not be generated as valid task data. "
            "No tasks were added. Please try again."
        )

    if schedule.clarification:
        return [], schedule.clarification

    valid_tasks = []
    seen = set()
    invalid_count = 0
    window_end = current_date + timedelta(days=window_days - 1)
    for raw_task in schedule.tasks:
        try:
            task = ScheduleTask.model_validate(raw_task)
        except ValidationError:
            invalid_count += 1
            continue
        if task.due_date < current_date or task.due_date > window_end:
            invalid_count += 1
            continue
        identity = (task.title.casefold(), task.due_date)
        if identity in seen:
            continue
        seen.add(identity)
        valid_tasks.append(task)

    if invalid_count:
        issue = (
            f"{invalid_count} task{'s' if invalid_count != 1 else ''} had dates "
            "with invalid or out-of-window details and were not added."
        )
        return valid_tasks, issue
    if _is_schedule_request(request.message) and not valid_tasks:
        return [], "No valid future tasks were generated, so nothing was added."
    return valid_tasks, None


@app.get("/")
def home():
    return {
        "project": "QuantaAI",
        "message": "QuantaAI backend is running!",
        "model": "QuantaLM",
        "teacher": "Online providers",
        "agents": AGENTS
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "FastAPI",
        "quantaLM_loaded": quanta_service.loaded,
        "teacher_available": teacher_service.is_available(),
        "online_provider_available": provider_manager.is_available(),
        "online_providers": provider_manager.online_provider_names(),
        "youtube_available": youtube_service.is_available(),
    }


@app.get("/agents")
def get_agents():
    return {
        "agents": AGENTS
    }


def _selected_provider(mode: str) -> str:
    try:
        return select_provider(mode)
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid QUANTAAI_AUTO_PROVIDER configuration."
                if mode == "auto"
                else "Invalid provider mode configuration."
            ),
        ) from None


def _fixed_conversational_reply(message: str) -> str | None:
    normalized = " ".join(message.casefold().split())
    normalized = re.sub(r"[.!?,]+$", "", normalized)
    greetings = {
        "hi": "Hey! How can I help you today?",
        "hello": "Hello! What can I help you with today?",
        "hey": "Hey! What can I help you with?",
        "good morning": "Good morning! What can I help you with today?",
        "good afternoon": "Good afternoon! What can I help you with today?",
        "good evening": "Good evening! What can I help you with today?",
        "how are you": "I'm doing well, thanks for asking! What can I help you with?",
    }
    if normalized in greetings:
        return greetings[normalized]
    if normalized in {"explain this topic in detail", "explain this in detail"}:
        return "What topic would you like me to explain in detail?"
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
    _csrf=Depends(require_csrf),
):
    current_date = _current_planner_date()
    if request.agent not in AGENTS:
        raise HTTPException(
            status_code=422,
            detail="Unknown agent selected."
        )

    if request.document_id is not None and request.agent != "Study":
        raise HTTPException(
            status_code=422,
            detail="PDF context is only available to the Study Agent.",
        )
    if request.conversation_id is not None and request.agent != "Surface":
        raise HTTPException(
            status_code=422,
            detail="Conversation PDF context is only available to Surface AI.",
        )
    document_context = (
        get_study_document_retrieval(
            db,
            user.id,
            request.document_id,
            request.message,
        )
        if request.document_id is not None
        else None
    )
    if request.agent == "Surface" and request.conversation_id is not None:
        attachments = conversation_document_attachments(
            db,
            request.conversation_id,
            user,
        )
        retrieved_documents = [
            get_study_document_retrieval(
                db,
                user.id,
                attachment["id"],
                request.message,
            )
            for attachment in attachments
            if isinstance(attachment, dict)
            and isinstance(attachment.get("id"), int)
        ]
        if retrieved_documents:
            document_context = StudyDocumentContext(
                prompt=(
                    "The user attached these PDFs to the current Surface AI "
                    "conversation. Use their retrieved excerpts as primary "
                    "evidence, distinguish document-supported details from "
                    "general knowledge, and do not invent facts or citations.\n\n"
                    + "\n\n".join(
                        document.prompt for document in retrieved_documents
                    )
                ),
                pages=tuple(
                    sorted({
                        page
                        for document in retrieved_documents
                        for page in document.pages
                    })
                ),
            )

    fixed_reply = _fixed_conversational_reply(request.message)
    if fixed_reply is not None:
        return ChatResponse(
            reply=fixed_reply,
            agent=request.agent,
            model="QuantaAI",
            provider="QuantaAI",
            status="success",
        )

    provider = _selected_provider(request.mode)
    history = [
        item.model_dump()
        for item in request.conversation_history
    ]
    if provider == "quanta":
        try:
            schedule_tasks = []
            schedule_tasks_error = None
            if request.agent == "Planner" and _is_schedule_request(request.message):
                schedule_tasks, schedule_tasks_error = _generate_schedule_tasks(
                    request=request,
                    history=history,
                    provider=provider,
                    current_date=current_date,
                )
                reply = (
                    _render_schedule_tasks(schedule_tasks)
                    if schedule_tasks
                    else schedule_tasks_error
                    or "No valid timetable tasks were generated, so nothing was added."
                )
            else:
                offline_system_prompt = None
                if request.agent == "Study":
                    offline_system_prompt = (
                        f"{AGENT_PROMPTS['Study']}\n\n"
                        f"{STUDY_TUTOR_MODE_PROMPTS[request.tutor_mode]}"
                    )
                    if document_context is not None:
                        offline_system_prompt = (
                            f"{offline_system_prompt}\n\n{document_context.prompt}"
                        )
                elif request.agent == "Surface" and document_context is not None:
                    offline_system_prompt = (
                        f"{AGENT_PROMPTS['Surface']}\n\n{document_context.prompt}"
                    )
                elif request.agent == "Planner":
                    offline_system_prompt = _planner_system_prompt(current_date)
                prompt = build_offline_prompt(
                    request.agent,
                    request.message,
                    conversation_history=history,
                    system_prompt=offline_system_prompt,
                )
                generated_reply = quanta_service.generate(
                    prompt,
                    max_new_tokens=(
                        900
                        if request.agent == "Planner"
                        else 160 if document_context is not None else 80
                    ),
                )
                if not isinstance(generated_reply, str) or not generated_reply.strip():
                    raise RuntimeError("QuantaLM returned an invalid response.")
                reply = generated_reply.strip()
            document_source = None
            document_pages = []
            if document_context is not None:
                reply, document_source, document_pages = resolve_document_answer(
                    reply,
                    document_context,
                )
            is_reliable = quanta_service.is_response_reliable(request.message, reply)
            if not is_reliable:
                logger.warning(
                    "QuantaLM response flagged by response-quality guard agent=%s",
                    request.agent,
                )
        except Exception as error:
            logger.error(
                "Offline QuantaLM generation failed agent=%s error_type=%s",
                request.agent,
                type(error).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Offline QuantaLM could not process the selected PDF context. "
                    "No online provider was used."
                    if document_context is not None
                    else "Offline QuantaLM generation failed."
                ),
            ) from error

        return ChatResponse(
            reply=reply,
            agent=request.agent,
            model="QuantaLM",
            provider="QuantaLM Offline",
            status="success" if is_reliable else "limited",
            experimental=True,
            reliability_warning=not is_reliable,
            document_source=document_source,
            document_pages=document_pages,
            schedule_tasks=schedule_tasks,
            schedule_tasks_error=schedule_tasks_error,
        )

    search_result = {"status": "skipped", "sources": [], "message": ""}
    if provider == "teacher" and should_search_web(request.message):
        search_result = web_search_service.search(
            request.message,
            deep=request.agent == "Research",
        )
    research_sources = search_result["sources"]
    research_context = build_research_context(research_sources)

    try:
        schedule_tasks = []
        schedule_tasks_error = None
        is_planner_schedule = (
            request.agent == "Planner" and _is_schedule_request(request.message)
        )
        if is_planner_schedule:
            schedule_tasks, schedule_tasks_error = _generate_schedule_tasks(
                request=request,
                history=history,
                provider=provider,
                current_date=current_date,
            )
            result = {
                "reply": (
                    _render_schedule_tasks(schedule_tasks)
                    if schedule_tasks
                    else schedule_tasks_error
                    or "No valid timetable tasks were generated, so nothing was added."
                ),
                "model": "Structured schedule generator",
                "provider": "Online provider",
            }
        elif request.agent == "Study":
            study_context = "\n\n".join(
                context
                for context in (
                    document_context.prompt if document_context is not None else None,
                    research_context,
                )
                if context
            ) or None
            result = study_agent.generate(
                request.message,
                conversation_history=history,
                document_context=study_context,
                tutor_mode=request.tutor_mode,
            )
        else:
            system_prompt = (
                _planner_system_prompt(current_date)
                if request.agent == "Planner"
                else AGENT_PROMPTS[request.agent]
            )
            if request.agent == "Surface" and document_context is not None:
                system_prompt = f"{system_prompt}\n\n{document_context.prompt}"
            if research_context:
                system_prompt = f"{system_prompt}{research_context}"
            result = provider_manager.generate(
                request.message,
                request.agent,
                history,
                system_prompt=system_prompt,
            )

        videos = []
        video_message = None
        if request.agent == "Study":
            try:
                query = provider_manager.generate_youtube_query(
                    request.message,
                    history,
                )
            except ProviderError as error:
                logger.warning(
                    "YouTube query generation failed category=%s",
                    error.category,
                )
                video_message = (
                    youtube_service.search("")["message"]
                    if not youtube_service.is_available()
                    else (
                        "A study answer is ready, but a YouTube search query could "
                        "not be prepared. Please try again."
                    )
                )
            else:
                if query:
                    video_result = youtube_service.search(query)
                    videos = video_result["videos"]
                    video_message = video_result["message"]

        reply = result["reply"]
        document_source = None
        document_pages = []
        if document_context is not None:
            reply, document_source, document_pages = resolve_document_answer(
                reply,
                document_context,
            )

        return ChatResponse(
            reply=reply,
            agent=request.agent,
            model=result["model"],
            provider=result["provider"],
            status="success",
            experimental=result["provider"] == "QuantaLM Offline",
            reliability_warning=(
                result.get("reliability") == "unreliable"
                and result["provider"] == "QuantaLM Offline"
            ),
            videos=videos,
            video_message=video_message,
            search_performed=bool(research_sources),
            search_agent=request.agent if research_sources else None,
            search_status=search_result["status"],
            search_message=search_result["message"],
            sources=[
                {
                    "id": source["id"],
                    "title": source["title"],
                    "url": source["url"],
                    "published_date": source["published_date"],
                }
                for source in research_sources
            ],
            document_source=document_source,
            document_pages=document_pages,
            schedule_tasks=schedule_tasks,
            schedule_tasks_error=schedule_tasks_error,
        )
    except ProviderUnavailableError as error:
        logger.warning(
            "Online provider chain exhausted agent=%s category=%s",
            request.agent,
            error.category,
        )
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from None
    except ProviderError as error:
        logger.warning(
            "Online provider failure agent=%s provider=%s category=%s http_status=%s",
            request.agent,
            error.provider,
            error.category,
            error.status_code,
        )
        raise HTTPException(status_code=502, detail=str(error)) from None
    except Exception as error:
        logger.error(
            "Online provider integration failure agent=%s error_type=%s",
            request.agent,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Online provider integration failed due to an internal error.",
        ) from None