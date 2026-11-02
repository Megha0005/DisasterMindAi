import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  Activity, Eye, EyeOff, Mail, Lock,
  ArrowRight, ShieldCheck
} from "lucide-react";

const API_URL = "http://localhost:5000/api";

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }

    try {
      setLoading(true);

      const response = await axios.post(`${API_URL}/auth/login`, {
        email: email.trim(),
        password
      });

      const { token, user } = response.data;

      localStorage.setItem("token", token);
      localStorage.setItem("user", JSON.stringify(user));

      navigate("/");
    } catch (err) {
      setError(
        err.response?.data?.message ||
        "Unable to connect to the server. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-brand-panel">
        <div className="auth-brand">
          <div className="auth-brand-icon"><Activity size={27} /></div>
          <div>
            <strong>DisasterMind <span>AI</span></strong>
            <small>INTELLIGENCE PLATFORM</small>
          </div>
        </div>

        <div className="auth-brand-content">
          <span className="intel-eyebrow">INTELLIGENCE THROUGH AI</span>
          <h1>Understand risk.<br /><span>Prepare for uncertainty.</span></h1>
          <p>
            Explore earthquake research, probabilistic modelling and
            synthetic disaster scenarios through one intelligent
            research workspace.
          </p>
          <div className="auth-feature">
            <ShieldCheck size={19} />
            <span>Research-driven disaster intelligence</span>
          </div>
        </div>

        <small className="auth-copyright">
          DISASTERMIND AI · RESEARCH PLATFORM
        </small>
      </section>

      <section className="auth-form-panel">
        <div className="auth-form-box">
          <div className="auth-mobile-brand">
            <Activity size={23} />DisasterMind AI
          </div>

          <span className="intel-eyebrow">WELCOME BACK</span>
          <h2>Sign in to your workspace</h2>
          <p className="auth-subtitle">
            Enter your credentials to access your research dashboard.
          </p>

          <form onSubmit={handleSubmit}>
            <label className="auth-label">Email address</label>
            <div className="auth-input-wrap">
              <Mail size={17} />
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <label className="auth-label">Password</label>
            <div className="auth-input-wrap">
              <Lock size={17} />
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="auth-eye"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>

            {error && <div className="auth-error">{error}</div>}

            <button
              type="submit"
              className="auth-submit"
              disabled={loading}
            >
              {loading ? "Signing in..." : "Sign in"}
              {!loading && <ArrowRight size={17} />}
            </button>
          </form>

          <p className="auth-switch">
            Don't have an account? <Link to="/register">Create an account</Link>
          </p>

          <button className="auth-back" onClick={() => navigate("/")}>
            ← Back to dashboard
          </button>

          <p className="auth-security">
            <ShieldCheck size={14} /> Secured with JWT authentication
          </p>
        </div>
      </section>
    </main>
  );
}
