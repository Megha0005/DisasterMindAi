import { useMemo, useState } from "react";
import {
  FileText,
  Download,
  BarChart3,
  CalendarDays,
  Database,
  Info,
} from "lucide-react";

const benchmarks = [
  {
    model: "Logistic Regression",
    roc_auc: 0.866198,
    pr_auc: 0.611981,
    precision: 0.56483,
    recall: 0.605638,
    f1: 0.584522,
  },
  {
    model: "MLP Neural Network",
    roc_auc: 0.8791,
    pr_auc: 0.6388,
    precision: 0.6165,
    recall: 0.57,
    f1: 0.5923,
  },
  {
    model: "HistGradientBoosting",
    roc_auc: 0.898668,
    pr_auc: 0.654764,
    precision: 0.620974,
    recall: 0.585353,
    f1: 0.602638,
  },
];

export default function Reports() {
  const [format, setFormat] = useState("csv");

  const csv = useMemo(() => {
    const header = Object.keys(benchmarks[0]).join(",");
    return [
      header,
      ...benchmarks.map((row) => Object.values(row).join(",")),
    ].join("\n");
  }, []);

  const download = () => {
    const isCSV = format === "csv";
    const content = isCSV
      ? csv
      : `DISASTERMIND AI - MODEL BENCHMARK REPORT\n\nEvaluation period: January 2023 - September 2025\nDataset: Held-out earthquake grid-month test set\nSamples: 143,616\n\n${benchmarks.map((row) => `${row.model}\nROC-AUC: ${row.roc_auc}\nPR-AUC: ${row.pr_auc}\nPrecision: ${row.precision}\nRecall: ${row.recall}\nF1: ${row.f1}\n`).join("\n")}\n\nDisclaimer: Retrospective research classifier, not an operational earthquake early-warning system. Scores are not calibrated probabilities.`;
    const blob = new Blob([content], {
      type: isCSV ? "text/csv" : "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `disastermind-benchmark-report.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="intel-page">
      <div className="intel-heading">
        <div>
          <span className="intel-eyebrow">RESEARCH DOCUMENTATION</span>
          <h1>Reports & Analytics</h1>
          <p>Review model evaluation results and export benchmark summaries.</p>
        </div>
        <span className="intel-tag">
          <FileText size={15} /> RESEARCH REPORTS
        </span>
      </div>

      <section className="report-overview-grid">
        <div className="intel-panel report-overview-card">
          <CalendarDays />
          <span>Test period</span>
          <strong>2023 to 2025</strong>
          <small>Jan 2023 to Sep 2025</small>
        </div>
        <div className="intel-panel report-overview-card">
          <Database />
          <span>Test samples</span>
          <strong>143,616</strong>
          <small>Held-out grid-month records</small>
        </div>
        <div className="intel-panel report-overview-card">
          <BarChart3 />
          <span>Champion ROC-AUC</span>
          <strong>89.87%</strong>
          <small>HistGradientBoosting</small>
        </div>
      </section>

      <section className="intel-panel">
        <div className="intel-section-heading">
          <div>
            <h3>Model benchmark report</h3>
            <p>Verified held-out test results</p>
          </div>
          <span className="intel-tag">3 MODELS</span>
        </div>
        <div className="intel-table-wrap">
          <table className="intel-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 score</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.roc_auc.toFixed(4)}</td>
                  <td>{row.pr_auc.toFixed(4)}</td>
                  <td>{row.precision.toFixed(4)}</td>
                  <td>{row.recall.toFixed(4)}</td>
                  <td>{row.f1.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="intel-panel report-download">
        <div className="intel-icon">
          <Download />
        </div>
        <div className="report-download-copy">
          <h3>Export benchmark report</h3>
          <p>
            Download the currently available benchmark metrics. These are
            research results, not a live forecast report.
          </p>
        </div>
        <select value={format} onChange={(e) => setFormat(e.target.value)}>
          <option value="csv">CSV spreadsheet</option>
          <option value="txt">Text report</option>
        </select>
        <button onClick={download}>
          <Download size={16} /> Download
        </button>
      </section>

      <section className="intel-panel model-note">
        <Info size={18} />
        <p>
          Reports currently contain verified model benchmark metrics. Live
          disaster event feeds, operational forecasts and generated PDF reports
          are not connected to this page.
        </p>
      </section>
    </main>
  );
}
