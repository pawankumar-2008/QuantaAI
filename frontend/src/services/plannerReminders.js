export const DEFAULT_PLANNER_REMINDER_TIME = "09:00";
export const PLANNER_TIME_ZONE = "Asia/Kolkata";

function reminderStorageKey(userId) {
  return `quantaaI.plannerReminders.${userId}`;
}

function readReminderState(userId) {
  if (!userId || typeof localStorage === "undefined") {
    return { times: {}, notified: [] };
  }
  const raw = localStorage.getItem(reminderStorageKey(userId));
  if (!raw) return { times: {}, notified: [] };

  const state = JSON.parse(raw);
  return {
    times: state.times && typeof state.times === "object" ? state.times : {},
    notified: Array.isArray(state.notified) ? state.notified : [],
  };
}

function writeReminderState(userId, state) {
  if (!userId || typeof localStorage === "undefined") {
    throw new Error("Planner reminder settings are unavailable in this browser.");
  }
  localStorage.setItem(reminderStorageKey(userId), JSON.stringify(state));
}

export function loadPlannerReminderTimes(userId) {
  return { ...readReminderState(userId).times };
}

export function savePlannerReminderTime(userId, taskId, time) {
  if (!/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(time)) {
    throw new Error("Choose a valid reminder time.");
  }
  const state = readReminderState(userId);
  state.times[String(taskId)] = time;
  writeReminderState(userId, state);
  return { ...state.times };
}

export function getAsiaKolkataDateTime(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: PLANNER_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(now);
  const values = Object.fromEntries(
    parts
      .filter((part) => part.type !== "literal")
      .map(({ type, value }) => [type, value]),
  );
  return {
    date: `${values.year}-${values.month}-${values.day}`,
    time: `${values.hour}:${values.minute}`,
  };
}

export function getUpcomingPlannerTasks(tasks, now = new Date(), limit = 6) {
  const today = getAsiaKolkataDateTime(now).date;
  return tasks
    .filter((task) => !task.completed && task.due_date >= today)
    .slice(0, limit);
}

export function takeDuePlannerReminders(userId, tasks, now = new Date()) {
  const state = readReminderState(userId);
  const current = getAsiaKolkataDateTime(now);
  const dueTasks = tasks.filter((task) => {
    if (task.completed || task.due_date !== current.date) return false;
    const reminderTime =
      state.times[String(task.id)] || DEFAULT_PLANNER_REMINDER_TIME;
    const identity = `${task.id}|${task.due_date}`;
    if (reminderTime > current.time || state.notified.includes(identity)) {
      return false;
    }
    state.notified.push(identity);
    return true;
  });

  if (dueTasks.length > 0) writeReminderState(userId, state);
  return dueTasks;
}
