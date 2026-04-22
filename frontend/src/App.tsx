// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useTradingStore } from './store/tradingStore';
import { LoginPage, RegisterPage } from './components/AuthPages';
import { Dashboard } from './components/Dashboard';
import { useWebSocket } from './hooks/useWebSocket';

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { auth } = useTradingStore();
  if (!auth.isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  useWebSocket();
  return (
    <Routes>
      <Route path="/login"    element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/"         element={<AuthGuard><Dashboard /></AuthGuard>} />
      <Route path="*"         element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
