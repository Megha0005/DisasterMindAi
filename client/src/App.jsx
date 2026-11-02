import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import RiskAnalysis from "./pages/RiskAnalysis";
import ScenarioGenerator from "./pages/ScenarioGenerator";
import AppLayout from "./components/layout/AppLayout";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />

        <Route
          path="/risk-analysis"
          element={
            <AppLayout>
              <RiskAnalysis />
            </AppLayout>
          }
        />

        <Route
          path="/scenario-generator"
          element={
            <AppLayout>
              <ScenarioGenerator />
            </AppLayout>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
