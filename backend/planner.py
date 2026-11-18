import logging
from datetime import date, datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.auth import Base, User, get_db, require_csrf, require_user


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/planner/tasks", tags=["planner"])


class PlannerTask(Base):
    __tablename__ = "planner_tasks"

    __table_args__ = (
        Index("ix_planner_tasks_user_due", "user_id", "due_date"),
        CheckConstraint(
            "length(trim(title)) > 0",
            name="ck_planner_tasks_title",
        ),
    )

    id: Mapped[int] = mapped_column("task_id", Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    completed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
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


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    due_date: date

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("Task title cannot be empty.")
        return title

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        description = value.strip() if value is not None else None
        return description or None


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    due_date: date | None = None
    completed: bool | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        description = value.strip() if value is not None else None
        return description or None

    @model_validator(mode="after")
    def validate_updates(self):
        if not self.model_fields_set:
            raise ValueError("Provide at least one task update.")
        for field in ("title", "due_date", "completed"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field.replace('_', ' ').capitalize()} cannot be empty.")
        return self


def _task_response(task: PlannerTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "due_date": task.due_date,
        "completed": task.completed,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _owned_task(db: Session, task_id: int, user: User) -> PlannerTask:
    task = db.scalar(
        select(PlannerTask).where(
            PlannerTask.id == task_id,
            PlannerTask.user_id == user.id,
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@router.get("")
def list_tasks(
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, list[dict[str, Any]]]:
    try:
        tasks = db.scalars(
            select(PlannerTask)
            .where(PlannerTask.user_id == user.id)
            .order_by(
                PlannerTask.completed,
                PlannerTask.due_date,
                PlannerTask.id,
            )
        ).all()
    except SQLAlchemyError as error:
        logger.error("Planner task listing failed error_type=%s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Planner task storage is unavailable.",
        ) from None
    return {"tasks": [_task_response(task) for task in tasks]}


@router.post(
    "",
    status_code=201,
    dependencies=[Depends(require_csrf)],
)
def create_task(
    body: TaskCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, Any]:
    task = PlannerTask(
        user_id=user.id,
        title=body.title,
        description=body.description,
        due_date=body.due_date,
    )
    try:
        db.add(task)
        db.commit()
        db.refresh(task)
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("Planner task creation failed error_type=%s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Planner task storage is unavailable.",
        ) from None
    return _task_response(task)


@router.patch(
    "/{task_id}",
    dependencies=[Depends(require_csrf)],
)
def update_task(
    task_id: int,
    body: TaskUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> dict[str, Any]:
    try:
        task = _owned_task(db, task_id, user)
        updates = body.model_dump(exclude_unset=True)
        for field, value in updates.items():
            setattr(task, field, value)
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(task)
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("Planner task update failed error_type=%s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Planner task storage is unavailable.",
        ) from None
    return _task_response(task)


@router.delete(
    "/{task_id}",
    status_code=204,
    dependencies=[Depends(require_csrf)],
)
def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> None:
    try:
        task = _owned_task(db, task_id, user)
        db.delete(task)
        db.commit()
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("Planner task deletion failed error_type=%s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Planner task storage is unavailable.",
        ) from None
