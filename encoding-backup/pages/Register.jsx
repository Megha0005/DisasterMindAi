import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  Activity, Eye, EyeOff, Mail, Lock,
  User, ArrowRight, ShieldCheck
} from "lucide-react";

const API_URL = "http://localhost:5000/api";

export default function Register() {
  const [showPassword, setShowPassword] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const update = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (form.name.trim().length < 2) {
      setError("Please enter your full name.");
      return;
    }

    if (form.password.length < 8) {
      setError("Password must contain at least 8 characters.");
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      setLoading(true);

      const response = await axios.post(`${API_URL}/auth/register`, {
        name: form.name.trim(),
        email: form.email.trim(),
        password: form.password
      });

      const { token, user } = response.data;

      localStorage.setItem("token", token);
      localStorage.setItem("user", JSON.stringify(user));

      navigate("/");
    } catch (err) {
      setError(
        err.response?.data?.message ||
        "Unable to create account. Please try again."
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
          <span className="intel-eyebrow">JOIN THE WORKSPACE</span>
          <h1>Research smarter.<br /><span>Build safer futures.</span></h1>
          <p>
            Create your research workspace and explore AI-powered
            earthquake analysis and synthetic scenario generation.
          </p>
          <div className="auth-feature">
            <ShieldCheck size={19} />
            <span>A unified disaster research platform</span>
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

          <span className="intel-eyebrow">GET STARTED</span>
          <h2>Create your account</h2>
          <p className="auth-subtitle">
            Set up your research workspace.
          </p>

          <form onSubmit={handleSubmit}>
            <label className="auth-label">Full name</label>
            <div className="auth-input-wrap">
              <User size={17} />
              <input
                name="name"
                placeholder="Your full name"
                value={form.name}
                onChange={update}
                required
              />
            </div>

            <label className="auth-label">Email address</label>
            <div className="auth-input-wrap">
              <Mail size={17} />
              <input
                name="email"
                type="email"
                placeholder="you@example.com"
                value={form.email}
                onChange={update}
                required
              />
            </div>

            <label className="auth-label">Password</label>
            <div className="auth-input-wrap">
              <Lock size={17} />
              <input
                name="password"
                type={showPassword ? "text" : "password"}
                placeholder="Minimum 8 characters"
                value={form.password}
                onChange={update}
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

            <label className="auth-label">Confirm password</label>
            <div className="auth-input-wrap">
              <Lock size={17} />
              <input
                name="confirmPassword"
                type={showPassword ? "text" : "password"}
                placeholder="Confirm your password"
                value={form.confirmPassword}
                onChange={update}
                required
              />
            </div>

            {error && <div className="auth-error">{error}</div>}

            <button
              type="submit"
              className="auth-submit"
              disabled={loading}
            >
              {loading ? "Creating account..." : "Create account"}
              {!loading && <ArrowRight size={17} />}
            </button>
          </form>

          <p className="auth-switch">
            Already have an account? <Link to="/login">Sign in</Link>
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
