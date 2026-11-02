import { useState } from "react";
import { Activity, MapPin, CalendarDays, Waves, AlertTriangle, LoaderCircle, ShieldCheck } from "lucide-react";
import { aiService } from "../services/aiService";

export default function RiskAnalysis() {
  const [form, setForm] = useState({
    latitude: "20.5",
    longitude: "79.2",
    year: "2019",
    month: "11",
    past_1m_count: "1",
    past_3m_count: "2",
    past_6m_count: "4"
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (key, value) => {
    setForm(previous => ({ ...previous, [key]: value }));
  };

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([key, value]) => [
          key,
          Number(value)
        ])
      );

      const data = await aiService.predictEarthquake(payload);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        "Prediction request failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const fields = [
    { key: "latitude", label: "Latitude", icon: MapPin, step: "0.1", min: -90, max: 90 },
    { key: "longitude", label: "Longitude", icon: MapPin, step: "0.1", min: -180, max: 180 },
    { key: "year", label: "Historical year", icon: CalendarDays, step: "1", min: 2000, max: 2100 },
    { key: "month", label: "Month", icon: CalendarDays, step: "1", min: 1, max: 12 },
    { key: "past_1m_count", label: "Past 1 month count", icon: Waves, step: "1", min: 0 },
    { key: "past_3m_count", label: "Past 3 months count", icon: Waves, step: "1", min: 0 },
    { key: "past_6m_count", label: "Past 6 months count", icon: Waves, step: "1", min: 0 }
  ];

  return (
    <div className="risk-page">
      <div className="risk-heading">
        <div>
          <div className="eyebrow">EARTHQUAKE INTELLIGENCE</div>
          <h1>Earthquake Risk Analysis</h1>
          <p>Evaluate next-calendar-month earthquake occurrence using the trained statistical model.</p>
        </div>
        <div className="risk-model-tag"><Activity size={15} /> HISTGRADIENT BOOSTING</div>
      </div>

      <div className="risk-layout">
        <section className="risk-form-card">
          <div className="risk-card-heading">
            <div className="risk-heading-icon"><MapPin size={19} /></div>
            <div><h2>Analysis parameters</h2><p>Enter geographical and historical information</p></div>
          </div>

          <form onSubmit={submit}>
            <div className="risk-fields">
              {fields.map(({ key, label, step, min, max }) => (
                <label className="risk-field" key={key}>
                  <span>{label}</span>
                  <input
                    type="number"
                    step={step}
                    min={min}
                    max={max}
                    required
                    value={form[key]}
                    onChange={event => update(key, event.target.value)}
                  />
                </label>
              ))}
            </div>

            <div className="risk-note">
              <ShieldCheck size={17} />
              <p>Historical count inputs must satisfy: 1-month ≤ 3-month ≤ 6-month count.</p>
            </div>

            <button className="risk-submit" type="submit" disabled={loading}>
              {loading ? <><LoaderCircle className="spin" size={17} /> Analyzing...</> : <><Activity size={17} /> Run earthquake analysis</>}
            </button>
          </form>
        </section>

        <section className="risk-result-card">
          <div className="risk-card-heading">
            <div className="risk-heading-icon"><Activity size={19} /></div>
            <div><h2>Analysis results</h2><p>Live response from the AI Engine</p></div>
          </div>

          {!result && !loading && !error && (
            <div className="risk-empty">
              <div className="risk-empty-icon"><Waves size={34} /></div>
              <h3>Awaiting analysis</h3>
              <p>Submit the parameters to receive a model classification and score.</p>
            </div>
          )}

          {loading && <div className="risk-empty"><LoaderCircle className="spin" size={32} /><p>Running trained model...</p></div>}

          {error && <div className="risk-error"><AlertTriangle size={20} /><div><strong>Request failed</strong><p>{error}</p></div></div>}

          {result && (
            <div className="prediction-result">
              <div className={`prediction-status ${result.prediction === 1 ? "positive" : "negative"}`}>
                {result.prediction === 1 ? "Occurrence indicated" : "Occurrence not indicated"}
              </div>
              <div className="prediction-score">
                <span>Model score</span>
                <strong>{(result.probability * 100).toFixed(2)}%</strong>
              </div>
              <div className="score-track"><span style={{ width: `${result.probability * 100}%` }} /></div>
              <div className="result-row"><span>Classification threshold</span><strong>{result.classification_threshold}</strong></div>
              <div className="result-row"><span>Model</span><strong>{result.model}</strong></div>
              <div className="result-row"><span>Prediction horizon</span><strong>{result.prediction_horizon}</strong></div>
              <div className="grid-result">
                <span>2{"\u00B0"} × 2{"\u00B0"} GRID CELL</span>
                <strong>{result.grid_cell.latitude_range[0]}{"\u00B0"}–{result.grid_cell.latitude_range[1]}{"\u00B0"} N/S</strong>
                <strong>{result.grid_cell.longitude_range[0]}{"\u00B0"}–{result.grid_cell.longitude_range[1]}{"\u00B0"} E/W</strong>
              </div>
              <div className="result-disclaimer">{result.disclaimer}</div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
