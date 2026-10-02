
import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Cpu,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  Save,
  Settings2,
  Target,
  BarChart3,
  Table2,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./model-lab.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const CLASSES = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
];

const INITIAL_CONFIG = {
  epochs: 5,
  batch_size: 64,
  learning_rate: 0.001,
  initial_labeled_size: 1000,
  seed: 42,
};

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (error) {
    throw new Error(
      "Unable to connect to the ORACLE API. Check whether the backend is running."
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;

    try {
      const body = await response.json();
      detail = body.detail || body.message || detail;
    } catch {
      // Retain fallback error message.
    }

    throw new Error(detail);
  }

  return response.json();
}

const valid = (value) =>
  value !== null &&
  value !== undefined &&
  value !== "" &&
  Number.isFinite(Number(value));

const pct = (value) =>
  valid(value) ? `${Number(value).toFixed(2)}%` : "—";

const loss = (value) =>
  valid(value) ? Number(value).toFixed(4) : "—";

function duration(seconds) {
  if (!valid(seconds)) return "—";

  const totalSeconds = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  return hours
    ? `${hours}h ${minutes}m ${remainingSeconds}s`
    : minutes
      ? `${minutes}m ${remainingSeconds}s`
      : `${remainingSeconds}s`;
}

function MetricCard({ label, value, description, icon: Icon }) {
  return (
    <article className="ml-metric-card">
      <div className="ml-metric-top">
        <span>{label}</span>
        <Icon size={18} />
      </div>

      <strong>{value}</strong>
      <small>{description}</small>
    </article>
  );
}

function StatusBadge({ state }) {
  const value = String(state || "idle").toLowerCase();

  const labels = {
    idle: "Ready",
    queued: "Queued",
    preparing: "Preparing",
    training: "Training",
    evaluating: "Evaluating",
    completed: "Completed",
    failed: "Failed",
  };

  return (
    <span className={`ml-status ml-status-${value}`}>
      <span className="ml-status-dot" />
      {labels[value] || value}
    </span>
  );
}

function EvaluationPanel({ evaluation }) {
  if (!evaluation) {
    return (
      <section className="ml-panel">
        <div className="ml-panel-heading">
          <h2>
            <Target size={18} /> Model evaluation
          </h2>
        </div>

        <div className="ml-chart-empty">
          Evaluation metrics are not available for this experiment.
          Completed runs created before evaluation was added will not contain
          these values. Start a new run to generate them.
        </div>
      </section>
    );
  }

  const report = evaluation.classification_report || {};

  const rows = CLASSES.map((name) => ({
    name,
    ...(report[name] || {}),
  }));

  const macro = report.macro_avg || {};
  const weighted = report.weighted_avg || {};

  const matrix = evaluation.confusion_matrix || [];

  const maxValue = Math.max(
    1,
    ...matrix.flat().map(Number)
  );

  return (
    <>
      <section className="ml-panel">
        <div className="ml-panel-heading">
          <div>
            <h2>
              <Target size={18} /> Evaluation overview
            </h2>
            <p>Metrics calculated on the held-out test set.</p>
          </div>

          <span className="ml-history-count">
            {Number(evaluation.test_samples || 0).toLocaleString()} samples
          </span>
        </div>

        <div className="ml-eval-metrics">
          <div>
            <span>Test accuracy</span>
            <strong>{pct(evaluation.test_accuracy)}</strong>
          </div>

          <div>
            <span>Macro precision</span>
            <strong>{pct(Number(macro.precision) * 100)}</strong>
          </div>

          <div>
            <span>Macro recall</span>
            <strong>{pct(Number(macro.recall) * 100)}</strong>
          </div>

          <div>
            <span>Macro F1-score</span>
            <strong>{pct(Number(macro.f1_score) * 100)}</strong>
          </div>
        </div>
      </section>

      <section className="ml-panel">
        <div className="ml-panel-heading">
          <div>
            <h2>
              <Table2 size={18} /> Classification report
            </h2>
            <p>Per-class precision, recall, F1-score, and support.</p>
          </div>
        </div>

        <div className="ml-table-wrap">
          <table className="ml-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>Support</th>
              </tr>
            </thead>

            <tbody>
              {rows.map((row) => (
                <tr key={row.name}>
                  <td>
                    <strong>{row.name}</strong>
                  </td>
                  <td>
                    {valid(row.precision)
                      ? pct(Number(row.precision) * 100)
                      : "—"}
                  </td>
                  <td>
                    {valid(row.recall)
                      ? pct(Number(row.recall) * 100)
                      : "—"}
                  </td>
                  <td>
                    {valid(row.f1_score)
                      ? pct(Number(row.f1_score) * 100)
                      : "—"}
                  </td>
                  <td>{valid(row.support) ? row.support : "—"}</td>
                </tr>
              ))}

              <tr className="ml-summary-row">
                <td>
                  <strong>Macro average</strong>
                </td>
                <td>
                  {valid(macro.precision)
                    ? pct(Number(macro.precision) * 100)
                    : "—"}
                </td>
                <td>
                  {valid(macro.recall)
                    ? pct(Number(macro.recall) * 100)
                    : "—"}
                </td>
                <td>
                  {valid(macro.f1_score)
                    ? pct(Number(macro.f1_score) * 100)
                    : "—"}
                </td>
                <td>{valid(macro.support) ? macro.support : "—"}</td>
              </tr>

              <tr className="ml-summary-row">
                <td>
                  <strong>Weighted average</strong>
                </td>
                <td>
                  {valid(weighted.precision)
                    ? pct(Number(weighted.precision) * 100)
                    : "—"}
                </td>
                <td>
                  {valid(weighted.recall)
                    ? pct(Number(weighted.recall) * 100)
                    : "—"}
                </td>
                <td>
                  {valid(weighted.f1_score)
                    ? pct(Number(weighted.f1_score) * 100)
                    : "—"}
                </td>
                <td>
                  {valid(weighted.support) ? weighted.support : "—"}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="ml-panel">
        <div className="ml-panel-heading">
          <div>
            <h2>
              <BarChart3 size={18} /> Confusion matrix
            </h2>
            <p>
              Rows represent actual classes; columns represent predicted
              classes.
            </p>
          </div>
        </div>

        {matrix.length > 0 ? (
          <div className="ml-confusion-wrap">
            <div className="ml-confusion-axis">
              Predicted class →
            </div>

            <div className="ml-confusion-grid">
              <div className="ml-confusion-y">
                Actual class ↓
              </div>

              <div className="ml-confusion-table-wrap">
                <table className="ml-confusion-table">
                  <thead>
                    <tr>
                      <th></th>

                      {CLASSES.map((name) => (
                        <th key={name} title={name}>
                          {name.slice(0, 3)}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody>
                    {matrix.map((row, i) => (
                      <tr key={CLASSES[i] || i}>
                        <th>{CLASSES[i] || `Class ${i}`}</th>

                        {row.map((value, j) => {
                          const intensity = Number(value) / maxValue;

                          return (
                            <td
                              key={`${i}-${j}`}
                              title={`${CLASSES[i]} predicted as ${CLASSES[j]}: ${value}`}
                              style={{
                                background:
                                  i === j
                                    ? `rgba(66, 203, 215, ${0.15 + intensity * 0.75})`
                                    : `rgba(155, 140, 255, ${intensity * 0.65})`,
                                color:
                                  intensity > 0.48
                                    ? "#fff"
                                    : "#c8cadd",
                              }}
                            >
                              {value}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <p className="ml-chart-footnote">
              Strong diagonal values indicate correct predictions.
              Off-diagonal values show class confusions.
            </p>
          </div>
        ) : (
          <div className="ml-chart-empty">
            Confusion matrix unavailable.
          </div>
        )}
      </section>
    </>
  );
}

export default function ModelLab() {
  const [config, setConfig] = useState(INITIAL_CONFIG);
  const [status, setStatus] = useState(null);
  const [experiments, setExperiments] = useState([]);

  const [selectedId, setSelectedId] = useState("");
  const [selectedExperiment, setSelectedExperiment] = useState(null);

  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // Changing this value triggers a fresh experiment-detail request.
  const [detailRefresh, setDetailRefresh] = useState(0);

  const loadData = useCallback(async () => {
    try {
      const [statusData, experimentData] = await Promise.all([
        request("/api/model-lab/status"),
        request("/api/model-lab/experiments"),
      ]);

      setStatus(statusData);
      setExperiments(experimentData.items || []);
      setError("");

      // Refresh inspection after the history has been updated.
      setDetailRefresh((current) => current + 1);
    } catch (err) {
      setError(err.message || "Unable to load Model Lab data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const isRunning = Boolean(status?.running);

  // Automatically refresh live status while training or evaluating.
  useEffect(() => {
    if (!isRunning) return undefined;

    const timer = window.setInterval(() => {
      loadData();
    }, 2000);

    return () => window.clearInterval(timer);
  }, [isRunning, loadData]);

  const currentEpoch = Number(
    status?.completed_epochs ?? status?.epoch ?? 0
  );

  const totalEpochs = Number(status?.total_epochs || 0);

  const progress = totalEpochs
    ? Math.min(100, (currentEpoch / totalEpochs) * 100)
    : 0;

  const chartData = useMemo(
    () =>
      (status?.history || []).map((item) => ({
        epoch: item.epoch,
        trainLoss: item.train_loss,
        validationLoss: item.validation_loss,
        trainAccuracy: item.train_accuracy,
        validationAccuracy: item.validation_accuracy,
      })),
    [status?.history]
  );

  const latestExperiment = experiments[0];

  // Load selected experiment details.
  // The refresh counter ensures this runs again after history updates.
  useEffect(() => {
    if (!selectedId) {
      setSelectedExperiment(null);
      setDetailLoading(false);
      return undefined;
    }

    let active = true;

    setDetailLoading(true);
    setSelectedExperiment(null);

    request(
      `/api/model-lab/experiments/${encodeURIComponent(selectedId)}`
    )
      .then((data) => {
        if (!active) return;

        const experiment = data?.experiment || data;

        if (!experiment || typeof experiment !== "object") {
          throw new Error("Invalid experiment details received.");
        }

        setSelectedExperiment(experiment);
        setError("");
      })
      .catch((err) => {
        if (!active) return;

        setSelectedExperiment(null);
        setError(`Could not load experiment: ${err.message}`);
      })
      .finally(() => {
        if (active) {
          setDetailLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [selectedId, detailRefresh]);

  function updateConfig(key, value) {
    setConfig((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function startTraining(event) {
    event.preventDefault();

    setStarting(true);
    setError("");
    setNotice("");

    try {
      const result = await request("/api/model-lab/train", {
        method: "POST",
        body: JSON.stringify(config),
      });

      const experimentId = result.experiment_id || "";

      setNotice(
        `Training started. Experiment: ${experimentId}`
      );

      // Select the newly created experiment immediately.
      setSelectedId(experimentId);
      setSelectedExperiment(null);

      // Refresh status and experiment history.
      await loadData();
    } catch (err) {
      setError(err.message || "Could not start training.");
    } finally {
      setStarting(false);
    }
  }

  const selected = selectedExperiment;
  const evaluation = selected?.evaluation || null;
  const selectedHistory = selected?.history || [];

  return (
    <motion.section
      className="ml-page"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <header className="ml-heading">
        <div>
          <div className="ml-eyebrow">
            <FlaskConical size={15} />
            EXPERIMENTAL WORKSPACE
          </div>

          <h1>Model Lab</h1>

          <p>
            Configure, train, and evaluate your CIFAR-10
            classification model.
          </p>
        </div>

        <div className="ml-heading-actions">
          <StatusBadge state={status?.state} />

          <button
            className="ml-button ml-button-secondary"
            onClick={loadData}
            disabled={loading}
          >
            <RefreshCw size={15} />
            Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="ml-alert ml-alert-error">
          <AlertCircle size={17} />
          {error}
        </div>
      )}

      {notice && (
        <div className="ml-alert ml-alert-success">
          <CheckCircle2 size={17} />
          {notice}
        </div>
      )}

      <div className="ml-metric-grid">
        <MetricCard
          label="Model"
          value="Baseline CNN"
          description="3 convolutional blocks"
          icon={Cpu}
        />

        <MetricCard
          label="Current epoch"
          value={`${currentEpoch} / ${totalEpochs || "—"}`}
          description="Completed training passes"
          icon={Activity}
        />

        <MetricCard
          label="Validation accuracy"
          value={pct(status?.validation_accuracy)}
          description="Current epoch"
          icon={Target}
        />

        <MetricCard
          label="Best validation"
          value={pct(status?.best_validation_accuracy)}
          description="Best checkpoint selection"
          icon={CheckCircle2}
        />
      </div>

      <div className="ml-main-grid">
        <section className="ml-panel">
          <div className="ml-panel-heading">
            <div>
              <h2>
                <Settings2 size={18} />
                Training configuration
              </h2>
              <p>Set hyperparameters for your next experiment.</p>
            </div>
          </div>

          <form
            onSubmit={startTraining}
            className="ml-config-form"
          >
            <label className="ml-field">
              <span>Epochs</span>

              <input
                type="number"
                min="1"
                max="100"
                value={config.epochs}
                onChange={(event) =>
                  updateConfig(
                    "epochs",
                    Number(event.target.value)
                  )
                }
                disabled={isRunning || starting}
                required
              />

              <small>
                Number of complete training passes.
              </small>
            </label>

            <label className="ml-field">
              <span>Batch size</span>

              <select
                value={config.batch_size}
                onChange={(event) =>
                  updateConfig(
                    "batch_size",
                    Number(event.target.value)
                  )
                }
                disabled={isRunning || starting}
              >
                {[8, 16, 32, 64, 128, 256, 512].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>

              <small>
                Samples processed per optimization step.
              </small>
            </label>

            <label className="ml-field">
              <span>Learning rate</span>

              <input
                type="number"
                min="0.000001"
                max="0.1"
                step="any"
                value={config.learning_rate}
                onChange={(event) =>
                  updateConfig(
                    "learning_rate",
                    event.target.value === ""
                      ? ""
                      : Number(event.target.value)
                  )
                }
                disabled={isRunning || starting}
                required
              />

              <small>Example: 0.001</small>
            </label>

            <label className="ml-field">
              <span>Initial labeled samples</span>

              <input
                type="number"
                min="100"
                max="45000"
                step="100"
                value={config.initial_labeled_size}
                onChange={(event) =>
                  updateConfig(
                    "initial_labeled_size",
                    Number(event.target.value)
                  )
                }
                disabled={isRunning || starting}
                required
              />

              <small>
                Number of training samples used.
              </small>
            </label>

            <label className="ml-field">
              <span>Random seed</span>

              <input
                type="number"
                min="0"
                max="4294967295"
                value={config.seed}
                onChange={(event) =>
                  updateConfig(
                    "seed",
                    Number(event.target.value)
                  )
                }
                disabled={isRunning || starting}
                required
              />

              <small>
                Supports reproducible data splitting.
              </small>
            </label>

            <div className="ml-config-summary">
              <strong>Fixed experiment settings</strong>

              <div>
                <span>Dataset</span>
                <b>CIFAR-10</b>
              </div>

              <div>
                <span>Architecture</span>
                <b>BaselineCNN</b>
              </div>

              <div>
                <span>Optimizer</span>
                <b>Adam</b>
              </div>

              <div>
                <span>Loss</span>
                <b>CrossEntropyLoss</b>
              </div>

              <div>
                <span>Validation split</span>
                <b>5,000 images</b>
              </div>
            </div>

            <button
              className="ml-button ml-button-primary ml-start-button"
              type="submit"
              disabled={
                isRunning ||
                starting ||
                loading ||
                !valid(config.learning_rate) ||
                Number(config.learning_rate) <= 0
              }
            >
              {starting || isRunning ? (
                <LoaderCircle
                  className="ml-spin"
                  size={17}
                />
              ) : (
                <Play size={17} />
              )}

              {isRunning
                ? "Training in progress"
                : starting
                  ? "Starting..."
                  : "Start training"}
            </button>

            {isRunning && (
              <p className="ml-form-hint">
                Wait for the current run to finish before
                starting another.
              </p>
            )}
          </form>
        </section>

        <div className="ml-right-column">
          <section className="ml-panel">
            <div className="ml-panel-heading">
              <div>
                <h2>
                  <Activity size={18} />
                  Live training progress
                </h2>
                <p>
                  Automatically refreshed during training.
                </p>
              </div>
            </div>

            <div className="ml-progress-meta">
              <strong>
                {currentEpoch} of {totalEpochs || "—"} epochs
              </strong>
              <span>{progress.toFixed(0)}%</span>
            </div>

            <div
              className="ml-progress-track"
              role="progressbar"
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="ml-progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="ml-live-grid">
              <div>
                <span>Training loss</span>
                <strong>{loss(status?.train_loss)}</strong>
              </div>

              <div>
                <span>Training accuracy</span>
                <strong>{pct(status?.train_accuracy)}</strong>
              </div>

              <div>
                <span>Validation loss</span>
                <strong>{loss(status?.validation_loss)}</strong>
              </div>

              <div>
                <span>Validation accuracy</span>
                <strong>
                  {pct(status?.validation_accuracy)}
                </strong>
              </div>
            </div>

            {status?.experiment_id && (
              <div className="ml-experiment-id">
                <span>Experiment ID</span>
                <code>{status.experiment_id}</code>
              </div>
            )}

            {status?.state === "completed" && (
              <div className="ml-completion">
                <CheckCircle2 size={18} />
                Training completed. Best checkpoint saved.
              </div>
            )}

            {status?.state === "failed" && (
              <div className="ml-alert ml-alert-error">
                <AlertCircle size={17} />
                {status.error || "Training failed."}
              </div>
            )}
          </section>

          {[
            {
              title: "Training vs validation loss",
              description:
                "Lower values indicate lower prediction error.",
              keys: [
                ["trainLoss", "Training loss", "#9b8cff"],
                [
                  "validationLoss",
                  "Validation loss",
                  "#42cbd7",
                ],
              ],
              domain: undefined,
            },
            {
              title: "Training vs validation accuracy",
              description:
                "Accuracy across completed epochs.",
              keys: [
                [
                  "trainAccuracy",
                  "Training accuracy",
                  "#9b8cff",
                ],
                [
                  "validationAccuracy",
                  "Validation accuracy",
                  "#42cbd7",
                ],
              ],
              domain: [0, 100],
            },
          ].map((chart) => (
            <section
              className="ml-panel"
              key={chart.title}
            >
              <div className="ml-panel-heading">
                <div>
                  <h2>{chart.title}</h2>
                  <p>{chart.description}</p>
                </div>
              </div>

              {chartData.length > 0 ? (
                <div className="ml-chart">
                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >
                    <LineChart
                      data={chartData}
                      margin={{
                        top: 10,
                        right: 12,
                        left: -15,
                        bottom: 4,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#2b2e40"
                      />

                      <XAxis
                        dataKey="epoch"
                        stroke="#858ba0"
                        tick={{ fontSize: 11 }}
                      />

                      <YAxis
                        domain={chart.domain}
                        stroke="#858ba0"
                        tick={{ fontSize: 11 }}
                        tickFormatter={
                          chart.domain
                            ? (value) => `${value}%`
                            : undefined
                        }
                      />

                      <Tooltip
                        contentStyle={{
                          background: "#171927",
                          border: "1px solid #393d52",
                          borderRadius: 8,
                          color: "#eee",
                        }}
                      />

                      <Legend />

                      {chart.keys.map(
                        ([key, name, color]) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            name={name}
                            stroke={color}
                            strokeWidth={2.5}
                            dot={{ r: 3 }}
                          />
                        )
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="ml-chart-empty">
                  Metrics will appear after the first epoch.
                </div>
              )}
            </section>
          ))}
        </div>
      </div>

      <section className="ml-panel ml-history-panel">
        <div className="ml-panel-heading">
          <div>
            <h2>
              <Clock size={18} />
              Experiment history
            </h2>
            <p>
              Previous runs saved in local experiment history.
            </p>
          </div>

          <span className="ml-history-count">
            {experiments.length} runs
          </span>
        </div>

        {loading ? (
          <div className="ml-chart-empty">
            Loading experiment history...
          </div>
        ) : experiments.length === 0 ? (
          <div className="ml-chart-empty">
            No experiments yet. Start your first training run.
          </div>
        ) : (
          <div className="ml-table-wrap">
            <table className="ml-table">
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Status</th>
                  <th>Epochs</th>
                  <th>Best epoch</th>
                  <th>Duration</th>
                  <th>Best validation</th>
                  <th>Test accuracy</th>
                  <th>Evaluation</th>
                </tr>
              </thead>

              <tbody>
                {experiments.map((exp) => (
                  <tr
                    key={exp.experiment_id}
                    className={
                      selectedId === exp.experiment_id
                        ? "ml-selected-row"
                        : ""
                    }
                    onClick={() =>
                      setSelectedId(exp.experiment_id)
                    }
                    title="Click to inspect this experiment"
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      <strong>{exp.experiment_id}</strong>
                      <small>
                        {exp.started_at
                          ? new Date(
                              exp.started_at
                            ).toLocaleString()
                          : "—"}
                      </small>
                    </td>

                    <td>
                      <StatusBadge state={exp.status} />
                    </td>

                    <td>
                      {exp.completed_epochs ??
                        exp.history?.length ??
                        0}{" "}
                      / {exp.config?.epochs ?? "—"}
                    </td>

                    <td>{exp.best_epoch ?? "—"}</td>

                    <td>
                      {duration(exp.duration_seconds)}
                    </td>

                    <td>
                      {pct(exp.best_validation_accuracy)}
                    </td>

                    <td>{pct(exp.test_accuracy)}</td>

                    <td>
                      {exp.evaluation
                        ? "Available"
                        : "Not recorded"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {latestExperiment?.checkpoint && (
          <div className="ml-checkpoint-note">
            <Save size={16} />
            Latest checkpoint:{" "}
            <code>{latestExperiment.checkpoint}</code>
          </div>
        )}
      </section>

      <div className="ml-panel ml-history-panel">
        <div className="ml-panel-heading">
          <div>
            <h2>
              <Target size={18} />
              Experiment inspection
            </h2>
            <p>
              Select a row in experiment history to view its
              evaluation.
            </p>
          </div>

          {selected && (
            <StatusBadge state={selected.status} />
          )}
        </div>

        {!selectedId ? (
          <div className="ml-chart-empty">
            Select an experiment from the history table.
          </div>
        ) : detailLoading ? (
          <div className="ml-chart-empty">
            <LoaderCircle className="ml-spin" />
            Loading experiment details...
          </div>
        ) : selected ? (
          <>
            <div className="ml-inspection-meta">
              <span>
                <b>Experiment:</b>{" "}
                {selected.experiment_id}
              </span>

              <span>
                <b>Seed:</b>{" "}
                {selected.config?.seed ?? "—"}
              </span>

              <span>
                <b>Best epoch:</b>{" "}
                {selected.best_epoch ?? "—"}
              </span>

              <span>
                <b>Test accuracy:</b>{" "}
                {pct(selected.test_accuracy)}
              </span>
            </div>

            <EvaluationPanel evaluation={evaluation} />

            {selectedHistory.length > 0 && (
              <section className="ml-panel ml-inner-panel">
                <div className="ml-panel-heading">
                  <div>
                    <h2>
                      Selected run training history
                    </h2>
                    <p>
                      Epoch-by-epoch recorded metrics.
                    </p>
                  </div>
                </div>

                <div className="ml-table-wrap">
                  <table className="ml-table">
                    <thead>
                      <tr>
                        <th>Epoch</th>
                        <th>Train loss</th>
                        <th>Train accuracy</th>
                        <th>Validation loss</th>
                        <th>Validation accuracy</th>
                      </tr>
                    </thead>

                    <tbody>
                      {selectedHistory.map((item) => (
                        <tr key={item.epoch}>
                          <td>{item.epoch}</td>
                          <td>{loss(item.train_loss)}</td>
                          <td>
                            {pct(item.train_accuracy)}
                          </td>
                          <td>
                            {loss(item.validation_loss)}
                          </td>
                          <td>
                            {pct(item.validation_accuracy)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </>
        ) : (
          <div className="ml-chart-empty">
            Experiment details are unavailable.
            {error && (
              <p>
                Check the error message displayed at the top
                of the page.
              </p>
            )}
          </div>
        )}
      </div>
    </motion.section>
  );
}