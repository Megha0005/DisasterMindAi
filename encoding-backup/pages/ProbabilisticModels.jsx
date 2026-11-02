import { useState } from "react";
import { BrainCircuit, Activity, Target, BarChart3, Info } from "lucide-react";

const models = [
  { name: "Logistic Regression", auc: 0.866198, pr: 0.611981, precision: 0.564830, recall: 0.605638, f1: 0.584522, threshold: 0.784315 },
  { name: "MLP Neural Network", auc: 0.8791, pr: 0.6388, precision: 0.6165, recall: 0.5700, f1: 0.5923, threshold: 0.78323 },
  { name: "HistGradientBoosting", auc: 0.898668, pr: 0.654764, precision: 0.620974, recall: 0.585353, f1: 0.602638, threshold: 0.8086 }
];

export default function ProbabilisticModels() {
  const [selected, setSelected] = useState("HistGradientBoosting");
  const model = models.find(item => item.name === selected);

  return (
    <main className="intel-page">
      <div className="intel-heading">
        <div>
          <span className="intel-eyebrow">MODEL EVALUATION</span>
          <h1>Probabilistic Models</h1>
          <p>Compare trained earthquake occurrence classifiers on held-out test data.</p>
        </div>
        <span className="intel-tag"><BrainCircuit size={15} /> 3 EVALUATED MODELS</span>
      </div>

      <section className="intel-panel model-note">
        <Info size={18} />
        <p>Evaluation period: January 2023 – September 2025. These metrics measure retrospective classification performance, not operational forecasting accuracy.</p>
      </section>

      <section className="model-cards">
        {models.map(item => (
          <button key={item.name} className={`intel-panel model-card ${selected === item.name ? "model-selected" : ""}`} onClick={() => setSelected(item.name)}>
            <div className="model-card-top"><BrainCircuit size={19} /><span>{selected === item.name ? "SELECTED" : "VIEW MODEL"}</span></div>
            <h3>{item.name}</h3>
            <strong>{(item.auc * 100).toFixed(2)}%</strong>
            <small>Test ROC-AUC</small>
            <div className="model-progress"><span style={{ width: `${item.auc * 100}%` }} /></div>
            {item.name === "HistGradientBoosting" && <span className="model-champion">Best test ROC-AUC and F1</span>}
          </button>
        ))}
      </section>

      <section className="intel-panel model-detail">
        <div className="intel-section-heading">
          <div><h3>{model.name}</h3><p>Held-out test performance</p></div>
          <span className="intel-tag">TEST SET</span>
        </div>
        <div className="model-metrics-grid">
          <div><Target /><span>ROC-AUC</span><strong>{model.auc.toFixed(4)}</strong></div>
          <div><BarChart3 /><span>PR-AUC</span><strong>{model.pr.toFixed(4)}</strong></div>
          <div><Activity /><span>Precision</span><strong>{model.precision.toFixed(4)}</strong></div>
          <div><Activity /><span>Recall</span><strong>{model.recall.toFixed(4)}</strong></div>
          <div><Target /><span>F1 score</span><strong>{model.f1.toFixed(4)}</strong></div>
          <div><Info /><span>Decision threshold</span><strong>{model.threshold.toFixed(4)}</strong></div>
        </div>
      </section>

      <section className="intel-panel">
        <div className="intel-section-heading"><div><h3>Model benchmark comparison</h3><p>All values from the same held-out test period</p></div></div>
        <div className="intel-table-wrap">
          <table className="intel-table">
            <thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>{models.map(item => <tr key={item.name}><td>{item.name}</td><td>{item.auc.toFixed(4)}</td><td>{item.pr.toFixed(4)}</td><td>{item.precision.toFixed(4)}</td><td>{item.recall.toFixed(4)}</td><td>{item.f1.toFixed(4)}</td></tr>)}</tbody>
          </table>
        </div>
      </section>

      <section className="intel-panel model-note">
        <Info size={18} />
        <p>Scores are not calibrated probabilities. The classifier predicts next-calendar-month M4+ event presence in a 2° × 2° grid cell. It does not predict an exact earthquake time, magnitude or epicentre.</p>
      </section>
    </main>
  );
}
