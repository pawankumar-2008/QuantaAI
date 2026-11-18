import { useState } from "react";
import {
  DEFAULT_PLANNER_REMINDER_TIME,
  getAsiaKolkataDateTime,
  PLANNER_TIME_ZONE,
} from "../services/plannerReminders";

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function toDateKey(date) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function formatDueDate(value) {
  if (!value) return "No due date";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(
    new Date(`${value}T00:00:00`),
  );
}

function Planner({
  tasks,
  loading,
  error,
  mutationId,
  reminderTimes,
  reminderError,
  browserNotificationsEnabled,
  onEnableBrowserNotifications,
  onReminderTimeChange,
  onCreate,
  onUpdate,
  onDelete,
  onBack,
}) {
  const today = getAsiaKolkataDateTime().date;
  const [title, setTitle] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDueDate, setEditDueDate] = useState("");
  const [selectedDate, setSelectedDate] = useState(today);
  const [selectedTaskId, setSelectedTaskId] = useState(null);
  const [calendarMonth, setCalendarMonth] = useState(
    () => new Date(`${today}T00:00:00`),
  );

  const tasksByDate = new Map();
  for (const task of tasks) {
    const datedTasks = tasksByDate.get(task.due_date) || [];
    datedTasks.push(task);
    tasksByDate.set(task.due_date, datedTasks);
  }
  const selectedDateTasks = tasksByDate.get(selectedDate) || [];
  const selectedTask =
    selectedDateTasks.find((task) => task.id === selectedTaskId) ||
    selectedDateTasks[0] ||
    null;

  const monthStart = new Date(
    calendarMonth.getFullYear(),
    calendarMonth.getMonth(),
    1,
  );
  const gridStart = new Date(
    monthStart.getFullYear(),
    monthStart.getMonth(),
    -monthStart.getDay() + 1,
  );
  const calendarDays = Array.from({ length: 42 }, (_, index) => {
    const date = new Date(gridStart);
    date.setDate(gridStart.getDate() + index);
    return date;
  });

  async function createTask(event) {
    event.preventDefault();
    const saved = await onCreate({ title: title.trim(), due_date: dueDate });
    if (saved) {
      setTitle("");
      setDueDate("");
    }
  }

  async function saveTask(event, taskId) {
    event.preventDefault();
    const saved = await onUpdate(taskId, {
      title: editTitle.trim(),
      due_date: editDueDate,
    });
    if (saved) setEditingId(null);
  }

  function changeMonth(offset) {
    setCalendarMonth(
      new Date(
        calendarMonth.getFullYear(),
        calendarMonth.getMonth() + offset,
        1,
      ),
    );
  }

  return (
    <section className="learning-page" aria-labelledby="planner-heading">
      <header className="learning-page-heading">
        <div>
          <span className="eyebrow">YOUR LEARNING PLAN</span>
          <h1 id="planner-heading">Planner &amp; To-do</h1>
          <p>Create, update, and track your learning tasks.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </header>

      <form className="learning-form planner-create-form" onSubmit={createTask}>
        <label>
          <span>Task</span>
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            maxLength={200}
            placeholder="Review recursion"
            required
          />
        </label>
        <label>
          <span>Due date</span>
          <input
            type="date"
            value={dueDate}
            onChange={(event) => setDueDate(event.target.value)}
            required
          />
        </label>
        <button type="submit" disabled={!title.trim() || !dueDate}>
          Add task
        </button>
      </form>

      {error && <p className="learning-error" role="alert">{error}</p>}
      {reminderError && (
        <p className="learning-error" role="alert">{reminderError}</p>
      )}

      <section className="planner-calendar" aria-labelledby="planner-calendar-heading">
        <header className="planner-calendar-header">
          <div>
            <h2 id="planner-calendar-heading">Schedule calendar</h2>
            <p>Marked dates contain tasks. Calendar reminders use {PLANNER_TIME_ZONE}.</p>
          </div>
          <div className="planner-calendar-controls">
            <button type="button" onClick={() => changeMonth(-1)} aria-label="Previous month">
              ‹
            </button>
            <strong>
              {new Intl.DateTimeFormat(undefined, {
                month: "long",
                year: "numeric",
              }).format(calendarMonth)}
            </strong>
            <button type="button" onClick={() => changeMonth(1)} aria-label="Next month">
              ›
            </button>
          </div>
        </header>

        <div className="planner-calendar-grid" role="grid" aria-label="Planner calendar">
          {WEEKDAYS.map((day) => (
            <span className="planner-calendar-weekday" role="columnheader" key={day}>
              {day}
            </span>
          ))}
          {calendarDays.map((date) => {
            const dateKey = toDateKey(date);
            const dayTasks = tasksByDate.get(dateKey) || [];
            const isSelected = selectedDate === dateKey;
            const isCurrentMonth = date.getMonth() === calendarMonth.getMonth();
            return (
              <button
                type="button"
                role="gridcell"
                key={dateKey}
                className={[
                  "planner-calendar-day",
                  isSelected ? "selected" : "",
                  !isCurrentMonth ? "outside-month" : "",
                  dateKey === today ? "today" : "",
                  dayTasks.length ? "has-tasks" : "",
                ].filter(Boolean).join(" ")}
                aria-label={`${dateKey}${dayTasks.length ? `, ${dayTasks.length} tasks` : ""}`}
                aria-pressed={isSelected}
                onClick={() => {
                  setSelectedDate(dateKey);
                  setSelectedTaskId(dayTasks[0]?.id ?? null);
                }}
              >
                <span>{date.getDate()}</span>
                {dayTasks.length > 0 && (
                  <small>{dayTasks.length} task{dayTasks.length === 1 ? "" : "s"}</small>
                )}
              </button>
            );
          })}
        </div>

        <div className="planner-calendar-selection">
          <h3>{formatDueDate(selectedDate)}</h3>
          {selectedDateTasks.length === 0 ? (
            <p>No tasks scheduled for this date.</p>
          ) : (
            <ul className="planner-calendar-day-tasks">
              {selectedDateTasks.map((task) => (
                <li key={task.id}>
                  <button
                    type="button"
                    className={selectedTask?.id === task.id ? "selected" : ""}
                    onClick={() => setSelectedTaskId(task.id)}
                  >
                    <strong>{task.title}</strong>
                    <span>{task.completed ? "Completed" : "Due"} {formatDueDate(task.due_date)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {selectedTask && (
            <article className="planner-selected-task">
              <h4>{selectedTask.title}</h4>
              <p>{selectedTask.description || "No task description provided."}</p>
              <time dateTime={selectedTask.due_date}>
                Due {formatDueDate(selectedTask.due_date)}
              </time>
              <label>
                <span>Reminder time ({PLANNER_TIME_ZONE})</span>
                <input
                  type="time"
                  value={
                    reminderTimes[String(selectedTask.id)] ||
                    DEFAULT_PLANNER_REMINDER_TIME
                  }
                  onChange={(event) =>
                    onReminderTimeChange(selectedTask.id, event.target.value)
                  }
                  aria-label={`Reminder time for ${selectedTask.title}`}
                />
              </label>
            </article>
          )}
          {typeof Notification !== "undefined" && (
            <div className="planner-notification-setting">
              <p>
                {browserNotificationsEnabled
                  ? "Browser notifications are enabled while this app is running."
                  : "In-app reminders work while QuantaAI is open. Browser notifications require permission and are not delivered while the browser is closed."}
              </p>
              {!browserNotificationsEnabled && (
                <button type="button" onClick={onEnableBrowserNotifications}>
                  Enable browser notifications
                </button>
              )}
            </div>
          )}
        </div>
      </section>

      <h2 className="planner-task-list-heading">All tasks</h2>
      {loading ? (
        <p className="learning-state" role="status">Loading your planner…</p>
      ) : tasks.length === 0 ? (
        <div className="learning-empty-state">
          <span aria-hidden="true">◷</span>
          <h2>No tasks yet</h2>
          <p>Add a learning task and due date to get started.</p>
        </div>
      ) : (
        <ul className="planner-task-list">
          {tasks.map((task) => {
            const isEditing = editingId === task.id;
            const isMutating = mutationId === String(task.id);
            return (
              <li
                className={`planner-task-card ${task.completed ? "completed" : ""}`}
                key={task.id}
              >
                <label className="planner-task-complete">
                  <input
                    type="checkbox"
                    checked={task.completed}
                    onChange={(event) =>
                      onUpdate(task.id, { completed: event.target.checked })
                    }
                    disabled={isMutating}
                    aria-label={`${task.completed ? "Reopen" : "Complete"} ${task.title}`}
                  />
                </label>
                {isEditing ? (
                  <form
                    className="planner-task-edit-form"
                    onSubmit={(event) => saveTask(event, task.id)}
                  >
                    <input
                      value={editTitle}
                      onChange={(event) => setEditTitle(event.target.value)}
                      maxLength={200}
                      aria-label="Task title"
                      required
                    />
                    <input
                      type="date"
                      value={editDueDate}
                      onChange={(event) => setEditDueDate(event.target.value)}
                      aria-label="Task due date"
                      required
                    />
                    <button
                      type="submit"
                      disabled={isMutating || !editTitle.trim() || !editDueDate}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      onClick={() => setEditingId(null)}
                      disabled={isMutating}
                    >
                      Cancel
                    </button>
                  </form>
                ) : (
                  <>
                    <div className="planner-task-copy">
                      <div className="planner-task-copy-details">
                        <strong>{task.title}</strong>
                        {task.description && <span>{task.description}</span>}
                      </div>
                      <time dateTime={task.due_date}>
                        Due {formatDueDate(task.due_date)}
                      </time>
                    </div>
                    <div className="planner-task-actions">
                      <button
                        type="button"
                        onClick={() => {
                          setEditingId(task.id);
                          setEditTitle(task.title);
                          setEditDueDate(task.due_date);
                        }}
                        disabled={isMutating}
                        aria-label={`Edit ${task.title}`}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(task.id)}
                        disabled={isMutating}
                        aria-label={`Delete ${task.title}`}
                      >
                        Delete
                      </button>
                    </div>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export default Planner;
