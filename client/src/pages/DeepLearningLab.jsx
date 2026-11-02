import { useState } from "react";
import { BrainCircuit, Activity, Info, CheckCircle2, AlertTriangle } from "lucide-react";

const experiments = [
  {
    id: "mlp", name: "Multilayer Perceptron", type: "Supervised classifier",
    status: "Evaluated", auc: "0.8791", pr: "0.6388",
    description: "Neural network classifier trained using a weighted binary cross-entropy objective.",
    findings: ["Test F1 score: 0.5923", "Test precision: 0.6165", "Test recall: 0.5700"],
    note: "Uses labelled next-month earthquake occurrence targets."
  },
  {
    id: "ae", name: "Autoencoder", type: "Unsupervised anomaly detection",
    status: "Evaluated", auc: null, pr: null,
    description: "Reconstructs input features through a three-dimensional latent representation.",
    findings: ["Test anomaly rate: 32.95%", "Test recall: 3.98%", "Test enrichment: 3.12�"],
    note: "Anomaly scores are descriptive reconstruction errors, not calibrated event probabilities."
  },
  {
    id: "vae", name: "Variational Autoencoder", type: "Unsupervised representation learning",
    status: "Evaluated", auc: null, pr: null,
    description: "Uses a three-dimensional latent representation with beta set to 0.1.",
    findings: ["Test anomaly rate: 14.73%", "Test recall: 23.63%", "Test enrichment: 1.604�"],
    note: "Reconstruction-based anomaly identification, not a future event classifier."
  },
  {
    id: "rbm", name: "Restricted Boltzmann Machine", type: "Generative experiment",
    status: "Not selected", auc: null, pr: null,
    description: "Experimental RBM attempts did not produce a sufficiently reliable generative model.",
    findings: ["No validated test benchmark", "Not used in the final pipeline"],
    note: "Excluded from the production workflow rather than reporting unsupported results."
  }
];

export default function DeepLearningLab() {
  const [active, setActive] = useState("mlp");
  const experiment = experiments.find(item => item.id === active);

  return (
    <main className="intel-page">
      <div className="intel-heading">
        <div>
          <span className="intel-eyebrow">NEURAL NETWORK RESEARCH</span>
          <h1>Deep Learning Lab</h1>
          <p>Explore trained neural networks and unsupervised learning experiments.</p>
        </div>
        <span className="intel-tag"><BrainCircuit size={15} /> EXPERIMENT WORKSPACE</span>
      </div>

      <section className="intel-panel model-note">
        <Info size={18} />
        <p>Experiment results shown here are from completed research runs. This interface does not launch new training jobs.</p>
      </section>

      <section className="experiment-grid">
        {experiments.map(item => (
          <button key={item.id} onClick={() => setActive(item.id)} className={`intel-panel experiment-card ${active === item.id ? "model-selected" : ""}`}>
            <div className="experiment-card-top">
              <BrainCircuit size={21} />
              <span className={item.status === "Evaluated" ? "experiment-status" : "experiment-status muted"}>{item.status}</span>
            </div>
            <h3>{item.name}</h3>
            <p>{item.type}</p>
          </button>
        ))}
      </section>

      <section className="intel-panel experiment-detail">
        <div className="intel-section-heading">
          <div><h3>{experiment.name}</h3><p>{experiment.type}</p></div>
          {experiment.status === "Evaluated" ? <CheckCircle2 color="#43c7ae" /> : <AlertTriangle color="#e4b65f" />}
        </div>
        <p className="experiment-description">{experiment.description}</p>

        {(experiment.auc || experiment.pr) && (
          <div className="model-metrics-grid">
            <div><Activity /><span>Test ROC-AUC</span><strong>{experiment.auc}</strong></div>
            <div><Activity /><span>Test PR-AUC</span><strong>{experiment.pr}</strong></div>
          </div>
        )}

        <h4>Experiment findings</h4>
        <ul className="experiment-findings">{experiment.findings.map(item => <li key={item}>{item}</li>)}</ul>
        <div className="model-note"><Info size={18} /><p>{experiment.note}</p></div>
      </section>

      <section className="intel-panel">
        <h3>Research interpretation</h3>
        <p className="intel-paragraph">The supervised MLP learns from labelled earthquake occurrence outcomes. The Autoencoder and VAE learn data representations without using the same supervised prediction objective. Their anomaly detection results should not be compared directly with the MLP classification metrics.</p>
      </section>
    </main>
  );
}
