import logging
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    DateTime,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.auth import Base, User, get_db, require_csrf, require_user
from backend.documents import StudyDocument


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])

AgentName = Literal[
    "Surface",
    "Study",
    "Coding",
    "Career",
    "Guide",
    "Planner",
    "Exam",
    "Research",
]
MessageAgent = Literal[
    "Surface",
    "Study",
    "Coding",
    "Career",
    "Guide",
    "Planner",
    "Exam",
    "Research",
    "Collaboration",
]


class DocumentAttachment(BaseModel):
    id: int = Field(gt=0)
    filename: str = Field(min_length=1, max_length=255)


class Conversation(Base):
    __tablename__ = "conversations"

    __table_args__ = (
        Index(
            "ix_conversations_user_updated",
            "user_id",
            "updated_at",
        ),
        CheckConstraint(
            "(conversation_type = 'surface' AND agent_id IS NULL) "
            "OR (conversation_type = 'agent' AND agent_id IS NOT NULL) "
            "OR (conversation_type = 'chat' AND agent_id IS NULL)",
            name="conversations_check",
        ),
        CheckConstraint(
            "conversation_type IN ('surface', 'agent', 'chat')",
            name="conversations_conversation_type_check",
        ),
    )

    id: Mapped[int] = mapped_column(
        "conversation_id",
        Integer,
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    conversation_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="chat",
    )

    agent_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        default="New chat",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ConversationMessage(Base):
    __tablename__ = "messages"

    __table_args__ = (
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "message_id",
        ),
    )

    id: Mapped[int] = mapped_column(
        "message_id",
        Integer,
        primary_key=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey(
            "conversations.conversation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        "sender_type",
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model: Mapped[str | None] = mapped_column(
        "model_key",
        String(200),
    )

    message_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    @property
    def agent(self) -> str | None:
        if isinstance(self.message_metadata, dict):
            return self.message_metadata.get("agent")

        return None


class ConversationCreate(BaseModel):
    title: str = Field(
        default="New chat",
        min_length=1,
        max_length=200,
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()

        if not title:
            raise ValueError("Title cannot be empty.")

        return title


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "error"]

    content: str = Field(
        min_length=1,
        max_length=20000,
    )

    agent: MessageAgent | None = None

    model: str | None = Field(
        default=None,
        max_length=200,
    )

    provider: str | None = Field(
        default=None,
        max_length=100,
    )

    experimental: bool = False

    reliability_warning: bool = False

    search_performed: bool = False

    search_agent: AgentName | None = None

    search_message: str = Field(default="", max_length=300)

    sources: list[dict[str, str]] = Field(default_factory=list, max_length=8)

    document_source: Literal[
        "uploaded_pdf",
        "pdf_ai_knowledge",
        "ai_knowledge",
    ] | None = None

    document_pages: list[int] = Field(default_factory=list, max_length=4)

    document_attachments: list[DocumentAttachment] = Field(
        default_factory=list,
        max_length=5,
    )

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        content = value.strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        return content

    @field_validator("document_pages")
    @classmethod
    def validate_document_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("Document page numbers must be positive.")
        return list(dict.fromkeys(value))


class MessagesCreate(BaseModel):
    messages: list[MessageCreate] = Field(
        min_length=1,
        max_length=50,
    )


def _conversation_response(
    conversation: Conversation,
) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def _message_response(
    message: ConversationMessage,
) -> dict[str, Any]:
    metadata = message.message_metadata

    role = (
        metadata.get("quantaai_role", message.role)
        if isinstance(metadata, dict)
        else message.role
    )

    return {
        "id": message.id,
        "role": role,
        "content": message.content,
        "agent": message.agent,
        "model": message.model,
        "provider": metadata.get("provider") if isinstance(metadata, dict) else None,
        "experimental": (
            metadata.get("experimental", False)
            if isinstance(metadata, dict)
            else False
        ),
        "reliability_warning": (
            metadata.get("reliability_warning", False)
            if isinstance(metadata, dict)
            else False
        ),
        "search_performed": (
            metadata.get("search_performed", False)
            if isinstance(metadata, dict)
            else False
        ),
        "search_agent": (
            metadata.get("search_agent")
            if isinstance(metadata, dict)
            else None
        ),
        "search_message": (
            metadata.get("search_message", "")
            if isinstance(metadata, dict)
            else ""
        ),
        "sources": metadata.get("sources", []) if isinstance(metadata, dict) else [],
        "document_source": (
            metadata.get("document_source")
            if isinstance(metadata, dict)
            else None
        ),
        "document_pages": (
            metadata.get("document_pages", [])
            if isinstance(metadata, dict)
            else []
        ),
        "document_attachments": (
            metadata.get("document_attachments", [])
            if isinstance(metadata, dict)
            else []
        ),
        "created_at": message.created_at,
    }


def conversation_document_attachments(
    db: Session,
    conversation_id: int,
    user: User,
) -> list[dict[str, Any]]:
    _owned_conversation(db, conversation_id, user)
    user_messages = db.scalars(
        select(ConversationMessage)
        .where(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.role == "user",
        )
        .order_by(ConversationMessage.id.desc())
    ).all()
    for message in user_messages:
        metadata = message.message_metadata
        if isinstance(metadata, dict) and metadata.get("agent") == "Surface":
            attachments = metadata.get("document_attachments", [])
            if not isinstance(attachments, list):
                return []
            document_ids = {
                attachment["id"]
                for attachment in attachments
                if isinstance(attachment, dict)
                and isinstance(attachment.get("id"), int)
            }
            if not document_ids:
                return []
            available_ids = set(
                db.scalars(
                    select(StudyDocument.id).where(
                        StudyDocument.user_id == user.id,
                        StudyDocument.id.in_(document_ids),
                        StudyDocument.status == "indexed",
                    )
                ).all()
            )
            return [
                attachment
                for attachment in attachments
                if isinstance(attachment, dict)
                and attachment.get("id") in available_ids
            ]
    return []


def _owned_conversation(
    db: Session,
    conversation_id: int,
    user: User,
) -> Conversation:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


@router.post(
    "",
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def create_conversation(
    body: ConversationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, Any]:

    conversation = Conversation(
        user_id=user.id,
        title=body.title,
        conversation_type="chat",
        status="active",
    )

    try:
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    except SQLAlchemyError as error:
        db.rollback()

        logger.error(
            "Conversation creation failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None

    return _conversation_response(conversation)


@router.get("")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, list[dict[str, Any]]]:

    try:
        conversations = db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.id.desc(),
            )
            .limit(limit)
        ).all()

    except SQLAlchemyError as error:
        logger.error(
            "Conversation listing failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None

    return {
        "conversations": [
            _conversation_response(item)
            for item in conversations
        ]
    }


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, Any]:

    try:
        conversation = _owned_conversation(
            db,
            conversation_id,
            user,
        )

        statement = select(ConversationMessage).where(
            ConversationMessage.conversation_id == conversation.id
        )

        if before_message_id is not None:
            statement = statement.where(
                ConversationMessage.id < before_message_id
            )

        rows = db.scalars(
            statement
            .order_by(ConversationMessage.id.desc())
            .limit(limit + 1)
        ).all()

    except HTTPException:
        raise

    except SQLAlchemyError as error:
        logger.error(
            "Conversation retrieval failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None

    has_more = len(rows) > limit
    messages = list(reversed(rows[:limit]))

    return {
        **_conversation_response(conversation),
        "messages": [
            _message_response(item)
            for item in messages
        ],
        "has_more": has_more,
    }


@router.patch(
    "/{conversation_id}",
    dependencies=[Depends(require_csrf)],
)
def rename_conversation(
    conversation_id: int,
    body: ConversationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, Any]:

    try:
        conversation = _owned_conversation(
            db,
            conversation_id,
            user,
        )

        conversation.title = body.title
        conversation.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(conversation)

    except HTTPException:
        raise

    except SQLAlchemyError as error:
        db.rollback()

        logger.error(
            "Conversation update failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None

    return _conversation_response(conversation)


@router.delete(
    "/{conversation_id}",
    status_code=204,
    dependencies=[Depends(require_csrf)],
)
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> None:

    try:
        conversation = _owned_conversation(
            db,
            conversation_id,
            user,
        )

        db.delete(conversation)
        db.commit()

    except HTTPException:
        raise

    except SQLAlchemyError as error:
        db.rollback()

        logger.error(
            "Conversation deletion failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None


@router.post(
    "/{conversation_id}/messages",
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def append_messages(
    conversation_id: int,
    body: MessagesCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, list[dict[str, Any]]]:

    try:
        conversation = _owned_conversation(
            db,
            conversation_id,
            user,
        )

        for item in body.messages:
            if not item.document_attachments:
                continue
            if item.role != "user" or item.agent != "Surface":
                raise HTTPException(
                    status_code=422,
                    detail="PDF attachments can only be saved to Surface user messages.",
                )
            for attachment in item.document_attachments:
                document = db.scalar(
                    select(StudyDocument).where(
                        StudyDocument.id == attachment.id,
                        StudyDocument.user_id == user.id,
                    )
                )
                if document is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Study document not found.",
                    )
                if attachment.filename != document.filename:
                    raise HTTPException(
                        status_code=422,
                        detail="The attachment filename does not match the stored PDF.",
                    )
                if document.status != "indexed":
                    raise HTTPException(
                        status_code=409,
                        detail="Only indexed PDFs can be attached to a conversation.",
                    )

        messages = [
            ConversationMessage(
                conversation_id=conversation.id,
                role=(
                    "assistant"
                    if item.role == "error"
                    else item.role
                ),
                content=item.content,
                message_metadata={
                    **(
                        {"agent": item.agent}
                        if item.agent
                        else {}
                    ),
                    **(
                        {"provider": item.provider}
                        if item.provider
                        else {}
                    ),
                    **(
                        {"experimental": True}
                        if item.experimental
                        else {}
                    ),
                    **(
                        {"reliability_warning": True}
                        if item.reliability_warning
                        else {}
                    ),
                    **(
                        {"search_performed": True}
                        if item.search_performed
                        else {}
                    ),
                    **(
                        {"search_agent": item.search_agent}
                        if item.search_agent
                        else {}
                    ),
                    **(
                        {"search_message": item.search_message}
                        if item.search_message
                        else {}
                    ),
                    **({"sources": item.sources} if item.sources else {}),
                    **(
                        {"document_source": item.document_source}
                        if item.document_source
                        else {}
                    ),
                    **(
                        {"document_pages": item.document_pages}
                        if item.document_pages
                        else {}
                    ),
                    **(
                        {
                            "document_attachments": [
                                {
                                    "id": attachment.id,
                                    "filename": attachment.filename,
                                }
                                for attachment in item.document_attachments
                            ]
                        }
                        if item.document_attachments
                        else {}
                    ),
                    **(
                        {"quantaai_role": "error"}
                        if item.role == "error"
                        else {}
                    ),
                },
                model=item.model,
            )
            for item in body.messages
        ]

        db.add_all(messages)

        conversation.updated_at = datetime.now(timezone.utc)

        db.commit()

        for message in messages:
            db.refresh(message)

    except HTTPException:
        raise

    except SQLAlchemyError as error:
        db.rollback()

        logger.error(
            "Conversation messages failed error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Conversation storage is unavailable.",
        ) from None

    return {
        "messages": [
            _message_response(item)
            for item in messages
        ]
    }