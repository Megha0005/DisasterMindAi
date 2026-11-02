import { useMemo, useState } from "react";
import {
  Activity, MapPin, Sparkles, Database, RefreshCw,
  AlertTriangle, BarChart3, Layers, Download
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip
} from "recharts";
import { aiService } from "../services/aiService";

export default function ScenarioGenerator() {
  const [form, setForm] = useState({
    scenario_name: "Earthquake Scenario",
    latitude: "20.5",
    longitude: "79.2",
    n_scenarios: "100"
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (event) => {
    setForm({ ...form, [event.target.name]: event.target.value });
  };

  const generate = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);

    const latitude = Number(form.latitude);
    const longitude = Number(form.longitude);
    const n = Number(form.n_scenarios);

    if (!form.scenario_name.trim() || form.scenario_name.trim().length < 3) {
      setError("Scenario name must contain at least 3 characters.");
      return;
    }

    if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
      setError("Latitude must be between -90 and 90.");
      return;
    }

    if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
      setError("Longitude must be between -180 and 180.");
      return;
    }

    if (!Number.isInteger(n) || n < 1 || n > 1000) {
      setError("Number of scenarios must be an integer between 1 and 1000.");
      return;
    }

    setLoading(true);

    try {
      const data = await aiService.generateScenarios({
        scenario_name: form.scenario_name.trim(),
        latitude,
        longitude,
        n_scenarios: n
      });
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        "Unable to generate scenarios. Check that all services are running."
      );
    } finally {
      setLoading(false);
    }
  };

  const records = result?.scenarios || [];

  const stats = useMemo(() => {
    if (!records.length) return null;

    const average = (key) =>
      records.reduce((sum, item) => sum + item[key], 0) / records.length;

    const maximum = (key) =>
      Math.max(...records.map((item) => item[key]));

    const allZero = records.filter(
      (item) => item.past_1m_count === 0 &&
        item.past_3m_count === 0 && item.past_6m_count === 0
    ).length;

    return {
      avg1: average("past_1m_count"),
      avg3: average("past_3m_count"),
      avg6: average("past_6m_count"),
      max6: maximum("past_6m_count"),
      allZero,
      unique: new Set(
        records.map((item) =>
          `${item.past_1m_count}-${item.past_3m_count}-${item.past_6m_count}`
        )
      ).size
    };
  }, [records]);

  const chartData = stats ? [
    { period: "Past 1 month", average: Number(stats.avg1.toFixed(3)) },
    { period: "Past 3 months", average: Number(stats.avg3.toFixed(3)) },
    { period: "Past 6 months", average: Number(stats.avg6.toFixed(3)) }
  ] : [];

  const downloadCSV = () => {
    if (!records.length) return;

    const columns = [
      "scenario_id", "scenario_type", "grid_lat", "grid_lon",
      "past_1m_count", "past_3m_count", "past_6m_count"
    ];

    const rows = records.map((item) =>
      columns.map((key) => item[key]).join(",")
    );

    const csv = [columns.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "disastermind-synthetic-scenarios.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="scenario-page">
      <div className="scenario-heading">
        <div>
          <span className="scenario-eyebrow">PROBABILISTIC MODELLING</span>
          <h1>Scenario Generator</h1>
          <p>Generate synthetic earthquake-count patterns using the V3 empirical model.</p>
        </div>
        <div className="scenario-model-tag">
          <Activity size={16} />
          V3 INCREMENT GENERATOR
        </div>
      </div>

      <section className="scenario-layout">
        <form className="scenario-panel scenario-form" onSubmit={generate}>
          <div className="scenario-panel-heading">
            <div className="scenario-icon"><Sparkles size={20} /></div>
            <div>
              <h3>Generation parameters</h3>
              <p>Configure your synthetic scenario batch</p>
            </div>
          </div>

          <div className="scenario-field full">
            <label>Scenario name</label>
            <input
              name="scenario_name"
              value={form.scenario_name}
              onChange={updateField}
              minLength={3}
              maxLength={100}
              required
            />
          </div>

          <div className="scenario-fields-grid">
            <div className="scenario-field">
              <label>Latitude</label>
              <input
                type="number"
                name="latitude"
                value={form.latitude}
                onChange={updateField}
                min="-90"
                max="90"
                step="any"
                required
              />
            </div>
            <div className="scenario-field">
              <label>Longitude</label>
              <input
                type="number"
                name="longitude"
                value={form.longitude}
                onChange={updateField}
                min="-180"
                max="180"
                step="any"
                required
              />
            </div>
          </div>

          <div className="scenario-field full">
            <label>Number of scenarios</label>
            <input
              type="number"
              name="n_scenarios"
              value={form.n_scenarios}
              onChange={updateField}
              min="1"
              max="1000"
              step="1"
              required
            />
            <small>Choose between 1 and 1,000 synthetic records.</small>
          </div>

          <div className="scenario-location-note">
            <MapPin size={17} />
            <span>The selected coordinates identify a 2° × 2° display grid. They do not condition the generator's sampling.</span>
          </div>

          {error && (
            <div className="scenario-error">
              <AlertTriangle size={17} />
              {error}
            </div>
          )}

          <button className="scenario-generate-btn" type="submit" disabled={loading}>
            {loading ? (
              <><RefreshCw size={17} className="scenario-spin" /> Generating scenarios...</>
            ) : (
              <><Sparkles size={17} /> Generate scenarios</>
            )}
          </button>

          <p className="scenario-limit">Generation uses historical empirical count increments. It does not forecast future events.</p>
        </form>

        <section className="scenario-panel scenario-output">
          <div className="scenario-panel-heading">
            <div className="scenario-icon"><Database size={20} /></div>
            <div>
              <h3>Generation results</h3>
              <p>Live response from the AI Engine</p>
            </div>
          </div>

          {!result ? (
            <div className="scenario-empty">
              <div className="scenario-empty-icon"><Layers size={34} /></div>
              <h3>Ready to generate</h3>
              <p>Configure your parameters and generate a batch to view its synthetic count patterns and statistics.</p>
            </div>
          ) : (
            <>
              <div className="scenario-result-banner">
                <span>GENERATION COMPLETED</span>
                <strong>{result.generated_scenarios} scenarios</strong>
              </div>

              <div className="scenario-meta">
                <div><span>Model</span><strong>{result.model}</strong></div>
                <div><span>Method</span><strong>Empirical joint sampling</strong></div>
                <div><span>Scenario</span><strong>{result.scenario_name}</strong></div>
                <div><span>Unique patterns</span><strong>{stats.unique}</strong></div>
              </div>

              <div className="scenario-grid-info">
                <span>SELECTED 2° × 2° GRID</span>
                <strong>{records[0]?.latitude_range?.[0]}°–{records[0]?.latitude_range?.[1]}° N/S</strong>
                <strong>{records[0]?.longitude_range?.[0]}°–{records[0]?.longitude_range?.[1]}° E/W</strong>
              </div>

              <div className="scenario-disclaimer">
                <AlertTriangle size={17} />
                <span>{result.disclaimer}</span>
              </div>
            </>
          )}
        </section>
      </section>

      {result && stats && (
        <>
          <section className="scenario-stats-grid">
            <div className="scenario-stat-card">
              <span>Generated records</span>
              <strong>{records.length.toLocaleString()}</strong>
              <small>Total synthetic samples</small>
            </div>
            <div className="scenario-stat-card">
              <span>Unique patterns</span>
              <strong>{stats.unique.toLocaleString()}</strong>
              <small>Distinct count combinations</small>
            </div>
            <div className="scenario-stat-card">
              <span>All-zero patterns</span>
              <strong>{((stats.allZero / records.length) * 100).toFixed(1)}%</strong>
              <small>{stats.allZero} records</small>
            </div>
            <div className="scenario-stat-card">
              <span>Maximum 6-month count</span>
              <strong>{stats.max6}</strong>
              <small>Across generated records</small>
            </div>
          </section>

          <section className="scenario-panel scenario-chart-panel">
            <div className="scenario-panel-heading">
              <div className="scenario-icon"><BarChart3 size={20} /></div>
              <div>
                <h3>Average generated counts</h3>
                <p>Mean synthetic count for each historical window</p>
              </div>
            </div>
            <div className="scenario-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 15, right: 15, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#26364a" />
                  <XAxis dataKey="period" stroke="#8da2bd" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#8da2bd" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ background: "#101c2c", border: "1px solid #304158", borderRadius: 8 }}
                    labelStyle={{ color: "#fff" }}
                  />
                  <Bar dataKey="average" name="Average count" fill="#43c7ae" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="scenario-panel scenario-table-panel">
            <div className="scenario-table-heading">
              <div className="scenario-panel-heading">
                <div className="scenario-icon"><Database size={20} /></div>
                <div>
                  <h3>Generated scenario records</h3>
                  <p>Showing up to 100 records from the generated batch</p>
                </div>
              </div>
              <button className="scenario-export-btn" onClick={downloadCSV} type="button">
                <Download size={16} /> Export CSV
              </button>
            </div>

            <div className="scenario-table-wrap">
              <table className="scenario-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Grid latitude</th>
                    <th>Grid longitude</th>
                    <th>Past 1 month</th>
                    <th>Past 3 months</th>
                    <th>Past 6 months</th>
                  </tr>
                </thead>
                <tbody>
                  {records.slice(0, 100).map((item) => (
                    <tr key={item.scenario_id}>
                      <td>#{item.scenario_id}</td>
                      <td>{item.grid_lat}°</td>
                      <td>{item.grid_lon}°</td>
                      <td>{item.past_1m_count}</td>
                      <td>{item.past_3m_count}</td>
                      <td>{item.past_6m_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {records.length > 100 && (
              <p className="scenario-table-footnote">Showing first 100 of {records.length} records. Export CSV to access the complete batch.</p>
            )}
          </section>
        </>
      )}
    </main>
  );
}
