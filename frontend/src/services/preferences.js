export function readPreference(key, allowedValues, fallback) {
  const savedValue = window.localStorage.getItem(key);
  return allowedValues.includes(savedValue) ? savedValue : fallback;
}

export function savePreference(key, value) {
  window.localStorage.setItem(key, value);
}
