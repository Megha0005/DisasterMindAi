import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import DisasterMap from "./pages/DisasterMap";
import RiskAnalysis from "./pages/RiskAnalysis";
import ScenarioGenerator from "./pages/ScenarioGenerator";
import DeepLearningLab from "./pages/DeepLearningLab";
import ProbabilisticModels from "./pages/ProbabilisticModels";
import Reports from "./pages/Reports";
import Login from "./pages/Login";
import Register from "./pages/Register";

import AppLayout from "./components/layout/AppLayout";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />

        <Route path="/disaster-map" element={<AppLayout><DisasterMap /></AppLayout>} />
        <Route path="/risk-analysis" element={<AppLayout><RiskAnalysis /></AppLayout>} />
        <Route path="/scenario-generator" element={<AppLayout><ScenarioGenerator /></AppLayout>} />
        <Route path="/deep-learning" element={<AppLayout><DeepLearningLab /></AppLayout>} />
        <Route path="/probabilistic-models" element={<AppLayout><ProbabilisticModels /></AppLayout>} />
        <Route path="/reports" element={<AppLayout><Reports /></AppLayout>} />

        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
