// src/lib/api.ts
import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_BASE,
    timeout: 10000,
    headers: { 'Content-Type': 'application/json' },
  });

  client.interceptors.request.use((config) => {
    const token = localStorage.getItem('quantail_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  let refreshing = false;
  client.interceptors.response.use(
    (r) => r,
    async (error: AxiosError) => {
      if (error.response?.status === 401 && !refreshing) {
        const refresh = localStorage.getItem('quantail_refresh');
        if (refresh && !error.config?.url?.includes('/auth/')) {
          refreshing = true;
          try {
            const { data } = await axios.post(`${API_BASE}/api/auth/refresh`, { refresh_token: refresh });
            localStorage.setItem('quantail_token', data.access_token);
            if (error.config) {
              error.config.headers.Authorization = `Bearer ${data.access_token}`;
              return client.request(error.config);
            }
          } catch {
            localStorage.clear();
            window.location.href = '/login';
          } finally {
            refreshing = false;
          }
        } else {
          localStorage.clear();
          window.location.href = '/login';
        }
      }
      return Promise.reject(error);
    }
  );
  return client;
}

export const api = createApiClient();

export const authApi = {
  login: (username: string, password: string) =>
    api.post<{ access_token: string; refresh_token: string; user: any }>('/api/auth/login', { username, password }),
  register: (data: { username: string; email: string; password: string; full_name: string; invite_code: string }) =>
    api.post<{ access_token: string; refresh_token: string; user: any }>('/api/auth/register', data),
  me: () => api.get<any>('/api/auth/me'),
  refresh: (refresh_token: string) => api.post<{ access_token: string }>('/api/auth/refresh', { refresh_token }),
};

export const sessionApi = {
  start:   (config: object) => api.post('/api/session/start', config),
  stop:    ()               => api.post('/api/session/stop'),
  status:  ()               => api.get('/api/session/status'),
  history: ()               => api.get('/api/sessions/history'),
};

export const dataApi = {
  getTrades:       (limit = 50)  => api.get(`/api/trades?limit=${limit}`),
  getPnlHistory:   (limit = 200) => api.get(`/api/pnl-history?limit=${limit}`),
  getPriceHistory: (limit = 200) => api.get(`/api/price-history?limit=${limit}`),
  exportTrades:    ()            => api.get('/api/export/trades'),
};

export const adminApi = {
  createInvite: (note: string, expires_days: number) => api.post('/api/admin/invites', { note, expires_days }),
  listInvites:  ()                                    => api.get('/api/admin/invites'),
  listUsers:    ()                                    => api.get('/api/admin/users'),
};
