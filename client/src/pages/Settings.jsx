import { useNavigate } from "react-router-dom";
import {
  Settings as SettingsIcon,
  User,
  Mail,
  ShieldCheck,
  Lock,
  LogOut,
  ArrowLeft
} from "lucide-react";

export default function Settings() {
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem("user") || "null");

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login", { replace: true });
  };

  return (
    <main className="intel-page">
      <div className="intel-heading">
        <div>
          <span className="intel-eyebrow">ACCOUNT MANAGEMENT</span>
          <h1>Settings</h1>
          <p>Manage your DisasterMind AI workspace and account.</p>
        </div>
        <button className="hero-button" onClick={() => navigate("/")}>
          <ArrowLeft size={16} /> Dashboard
        </button>
      </div>

      <section className="intel-panel" style={{ padding: 24, marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 22 }}>
          <User size={22} />
          <div>
            <h2>Profile information</h2>
            <p className="intel-muted">Your registered account details</p>
          </div>
        </div>

        <div className="map-detail-item">
          <span><User size={15} /> Full name</span>
          <strong>{user?.name || "Guest User"}</strong>
        </div>

        <div className="map-detail-item">
          <span><Mail size={15} /> Email address</span>
          <strong>{user?.email || "Not available"}</strong>
        </div>
      </section>

      <section className="intel-panel" style={{ padding: 24, marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
          <ShieldCheck size={22} />
          <div>
            <h2>Security</h2>
            <p className="intel-muted">Authentication and account protection</p>
          </div>
        </div>

        <div className="map-detail-item">
          <span><Lock size={15} /> Authentication</span>
          <strong>JWT enabled</strong>
        </div>

        <div className="map-detail-item">
          <span>Password protection</span>
          <strong>bcrypt hashing enabled</strong>
        </div>

        <p className="intel-muted">
          Password changes and password recovery are not yet implemented.
        </p>
      </section>

      <section className="intel-panel" style={{ padding: 24 }}>
        <h2>Account actions</h2>
        <p className="intel-muted">
          Sign out of your DisasterMind AI account on this browser.
        </p>
        <button
          className="profile-logout"
          style={{ maxWidth: 180, marginTop: 16 }}
          onClick={handleLogout}
        >
          <LogOut size={17} /> Logout
        </button>
      </section>
    </main>
  );
}
