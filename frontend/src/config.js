const configuredApiBaseUrl = import.meta.env?.VITE_API_BASE_URL?.trim();
const isProductionBuild = import.meta.env?.PROD === true;

if (isProductionBuild && !configuredApiBaseUrl) {
  throw new Error("VITE_API_BASE_URL must be set for production builds.");
}

const defaultApiBaseUrl = import.meta.env?.DEV ? "http://localhost:8000" : "";

export const API_BASE_URL = (
  configuredApiBaseUrl || defaultApiBaseUrl
).replace(/\/+$/, "");
