import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import {
  DEFAULT_PLANNER_REMINDER_TIME,
  getAsiaKolkataDateTime,
  getUpcomingPlannerTasks,
  loadPlannerReminderTimes,
  savePlannerReminderTime,
  takeDuePlannerReminders,
} from "../src/services/plannerReminders.js";

const originalLocalStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage");

function installStorage() {
  const values = new Map();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, String(value)),
    },
  });
}

afterEach(() => {
  if (originalLocalStorage) {
    Object.defineProperty(globalThis, "localStorage", originalLocalStorage);
  } else {
    delete globalThis.localStorage;
  }
});

test("uses Asia/Kolkata date and default reminder time", () => {
  assert.deepEqual(
    getAsiaKolkataDateTime(new Date("2026-11-12T19:00:00.000Z")),
    { date: "2026-11-13", time: "00:30" },
  );
  assert.equal(DEFAULT_PLANNER_REMINDER_TIME, "09:00");
});

test("persists configurable reminders and consumes each due task only once", () => {
  installStorage();
  const task = {
    id: 21,
    title: "Daily PYQs",
    due_date: "2026-11-13",
    completed: false,
  };
  const nowAtTenFifteen = new Date("2026-11-13T04:45:00.000Z");
  const times = savePlannerReminderTime("user-5", 21, "10:15");
  assert.equal(times["21"], "10:15");
  assert.deepEqual(loadPlannerReminderTimes("user-5"), { "21": "10:15" });

  assert.deepEqual(takeDuePlannerReminders(
    "user-5",
    [task],
    new Date("2026-11-13T04:44:00.000Z"),
  ), []);
  assert.deepEqual(takeDuePlannerReminders("user-5", [task], nowAtTenFifteen), [task]);
  assert.deepEqual(takeDuePlannerReminders("user-5", [task], nowAtTenFifteen), []);
});

test("skips reminders for completed, deleted, and differently dated tasks", () => {
  installStorage();
  const tasks = [
    { id: 1, title: "Completed", due_date: "2026-11-13", completed: true },
    { id: 2, title: "Tomorrow", due_date: "2026-11-14", completed: false },
  ];

  assert.deepEqual(
    takeDuePlannerReminders(
      "user-8",
      tasks,
      new Date("2026-11-13T04:00:00.000Z"),
    ),
    [],
  );
});

test("upcoming task projection includes saved future tasks and excludes completed or past tasks", () => {
  const tasks = [
    { id: 1, title: "Today", due_date: "2026-11-13", completed: false },
    { id: 2, title: "Past", due_date: "2026-11-12", completed: false },
    { id: 3, title: "Done", due_date: "2026-11-14", completed: true },
    { id: 4, title: "Next day", due_date: "2026-11-14", completed: false },
  ];

  assert.deepEqual(
    getUpcomingPlannerTasks(tasks, new Date("2026-11-12T19:00:00.000Z")),
    [tasks[0], tasks[3]],
  );
});
