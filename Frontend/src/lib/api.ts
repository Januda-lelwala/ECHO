export const API_BASE: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const readJsonResponse = async <T>(response: Response): Promise<T> => {
  const body = await response.text();

  if (!body) {
    throw new Error(`API returned an empty response (HTTP ${response.status})`);
  }

  try {
    return JSON.parse(body) as T;
  } catch {
    throw new Error(`API returned a non-JSON response (HTTP ${response.status})`);
  }
};
