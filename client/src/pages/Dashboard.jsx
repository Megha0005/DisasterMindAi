import { useState } from "react";
import {
  Activity, LayoutDashboard, Map, ShieldAlert, Sparkles,
  BrainCircuit, FileText, Settings, Bell, Search,
  ArrowUpRight, Radio, Database, Cpu, ChevronRight, LogOut
} from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend
} from "recharts";

const navigation = [
  { name: "Overview", icon: LayoutDashboard, path: "/" },
  { name: "Disaster Map", icon: Map, path: "/disaster-map" },
  { name: "Risk Analysis", icon: ShieldAlert, path: "/risk-analysis" },
  { name: "Scenario Generator", icon: Sparkles, path: "/scenario-generator" },
  { name: "Deep Learning Lab", icon: BrainCircuit, path: "/deep-learning" },
  { name: "Reports", icon: FileText, path: "/reports" }
];

const modelData = [
  { model: "Logistic Regression", roc: 0.8662, pr: 0.6120 },
  { model: "MLP Neural Network", roc: 0.8791, pr: 0.6388 },
  { model: "Hist. Gradient Boosting", roc: 0.8987, pr: 0.6548 }
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [profileOpen, setProfileOpen] = useState(false);

  const user = JSON.parse(localStorage.getItem("user") || "null");

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login", { replace: true });
  };

  const metrics = [
    { label: "Test ROC-AUC", value: "89.87%", detail: "HistGradientBoosting", icon: Activity },
    { label: "Test PR-AUC", value: "65.48%", detail: "Held-out test data", icon: Radio },
    { label: "Test F1 Score", value: "60.26%", detail: "Validation-selected threshold", icon: ShieldAlert },
    { label: "Test Samples", value: "143,616", detail: "Jan 2023 – Sep 2025", icon: Database }
  ];

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
          <button className="nav-item settings-item" onClick={() => navigate("/settings")}>
            <Settings size={18} /><span>Settings</span>
          </button>
          <button className="profile sidebar-profile" onClick={() => setProfileOpen(!profileOpen)}>
            <div className="avatar">{user?.name?.charAt(0)?.toUpperCase() || "U"}</div>
            <div>
              <strong>{user?.name || "Guest User"}</strong>
              <small>{user?.email || "Not signed in"}</small>
            </div>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>Overview</strong></div>
          <div className="top-actions">
            <div className="engine-status"><span className="status-dot" /> AI Engine configured</div>
            <button className="icon-button" aria-label="Search"><Search size={18} /></button>
            <button className="icon-button" aria-label="Notifications"><Bell size={18} /></button>
            <div className="profile-menu">
              <button
                className="top-avatar"
                aria-label="Open profile menu"
                onClick={() => setProfileOpen(!profileOpen)}
              >
                {user?.name?.charAt(0)?.toUpperCase() || "U"}
              </button>

              {profileOpen && (
                <div className="profile-dropdown">
                  <div className="profile-dropdown-info">
                    <strong>{user?.name || "Guest User"}</strong>
                    <small>{user?.email || "Not signed in"}</small>
                  </div>
                  <button onClick={handleLogout} className="profile-logout">
                    <LogOut size={16} /> Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <div className="dashboard-content">
          <section className="welcome-section">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> DISASTER INTELLIGENCE</div>
              <h1>Disaster Intelligence <span>Overview</span></h1>
              <p>Monitor model performance and explore natural disaster analysis tools.</p>
            </div>
            <div className="date-chip"><Activity size={15} /> MODEL BENCHMARKS</div>
          </section>

          <section className="hero-panel">
            <div className="hero-copy">
              <div className="hero-tag"><Sparkles size={14} /> AI-POWERED ANALYSIS</div>
              <h2>Understand risk.<br /><span>Prepare for impact.</span></h2>
              <p>Explore earthquake activity patterns, evaluate predictive models and generate synthetic disaster scenarios.</p>
              <button className="hero-button" onClick={() => navigate("/risk-analysis")}>
                Explore risk analysis <ArrowUpRight size={17} />
              </button>
            </div>
            <div className="hero-visual">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="globe"><Activity size={58} strokeWidth={1.1} /></div>
              <div className="visual-label label-top">EARTH SYSTEMS</div>
              <div className="visual-label label-bottom">DATA-DRIVEN INSIGHTS</div>
              <div className="visual-coordinate">20.50{"\u00B0"} N<br />79.20{"\u00B0"} E</div>
            </div>
          </section>

          <section className="section-heading">
            <div><h2>Model performance</h2><p>Held-out test set · January 2023 – September 2025</p></div>
            <span className="benchmark-tag">BENCHMARK RESULTS</span>
          </section>

          <section className="metrics-grid">
            {metrics.map(({ label, value, detail, icon: Icon }) => (
              <article className="metric-card" key={label}>
                <div className="metric-top"><span>{label}</span><div className="metric-icon"><Icon size={18} /></div></div>
                <h3>{value}</h3>
                <p>{detail}</p>
              </article>
            ))}
          </section>

          <section className="chart-section">
            <div className="chart-card">
              <div className="chart-header">
                <div><h2>Model comparison</h2><p>ROC-AUC and PR-AUC on held-out test data</p></div>
                <span className="chart-indicator"><span /> TEST RESULTS</span>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={modelData} margin={{ top: 12, right: 12, left: -18, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#253044" />
                    <XAxis dataKey="model" tick={{ fill: "#98a6ba", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 1]} tick={{ fill: "#98a6ba", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#111b2c", border: "1px solid #29384d", borderRadius: 10, color: "#fff" }} />
                    <Legend />
                    <Bar dataKey="roc" name="ROC-AUC" fill="#43c6ac" radius={[5, 5, 0, 0]} />
                    <Bar dataKey="pr" name="PR-AUC" fill="#688cff" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="insight-card">
              <div className="insight-icon"><BrainCircuit size={20} /></div>
              <span className="insight-label">CURRENT CHAMPION</span>
              <h3>HistGradient<br />Boosting</h3>
              <p>Best held-out test performance among the three evaluated supervised models.</p>
              <div className="insight-stat"><span>ROC-AUC</span><strong>0.8987</strong></div>
              <div className="insight-stat"><span>F1 Score</span><strong>0.6026</strong></div>
              <div className="insight-foot"><span className="status-dot" /> Model artifact available</div>
            </div>
          </section>

          <footer className="dashboard-footer">
            <span>DisasterMind AI · Research & Analysis Platform</span>
            <span>Retrospective model benchmarks · Not an operational warning system</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

