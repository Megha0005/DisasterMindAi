import { NavLink, useNavigate } from "react-router-dom";
import {
  Activity, LayoutDashboard, Map, ShieldAlert,
  Sparkles, BrainCircuit, FileText, Settings,
  Bell, Search, ChevronRight
} from "lucide-react";

const navigation = [
  { name: "Overview", icon: LayoutDashboard, path: "/" },
  { name: "Disaster Map", icon: Map, path: "/disaster-map" },
  { name: "Risk Analysis", icon: ShieldAlert, path: "/risk-analysis" },
  { name: "Scenario Generator", icon: Sparkles, path: "/scenario-generator" },
  { name: "Deep Learning Lab", icon: BrainCircuit, path: "/deep-learning" },
  { name: "Reports", icon: FileText, path: "/reports" }
];

export default function AppLayout({ children }) {
  const navigate = useNavigate();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={23} /></div>
          <div>
            <h2>DisasterMind<span> AI</span></h2>
            <p>INTELLIGENCE PLATFORM</p>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map(({ name, icon: Icon, path }) => (
            <NavLink
              key={name}
              to={path}
              end={path === "/"}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
            >
              <Icon size={18} />
              <span>{name}</span>
              <ChevronRight size={15} className="nav-arrow" />
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="system-heading">
              <span className="status-dot" /> AI ENGINE
              <span className="online-text">READY</span>
            </div>
            <p>Prediction services configured</p>
            <div className="system-track"><span /></div>
          </div>

          <button
            className="nav-item settings-item"
            onClick={() => navigate("/settings")}
          >
            <Settings size={18} /><span>Settings</span>
          </button>

          <div className="profile">
            <div className="avatar">DM</div>
            <div>
              <strong>DisasterMind</strong>
              <small>Research Workspace</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>
            <strong>DisasterMind AI</strong>
          </div>

          <div className="top-actions">
            <div className="engine-status">
              <span className="status-dot" /> AI Engine configured
            </div>
            <button className="icon-button" aria-label="Search">
              <Search size={18} />
            </button>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <div className="top-avatar">DM</div>
          </div>
        </header>

        {children}
      </main>
    </div>
  );
}
