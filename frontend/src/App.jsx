
import "./dataset-explorer.css";
import ModelLab from "./ModelLab";

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";

import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  Database,
  LayoutDashboard,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Save,
  Target,
  FlaskConical,
  ChartNoAxesCombined,
  Images,
  Search,
  Filter,
} from "lucide-react";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
} from "recharts";

const API =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;

    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Use fallback error.
    }

    throw new Error(detail);
  }

  return response.json();
}

/* =========================================================
   ANNOTATION STUDIO
========================================================= */

function AnnotationStudio() {
  const [index, setIndex] = useState(0);
  const [image, setImage] = useState("");
  const [selected, setSelected] = useState("");
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [loadingImage, setLoadingImage] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadHistory = useCallback(async () => {
    const data = await request("/api/annotations");
    setHistory(data.items || []);
    setTotal(data.total || 0);
  }, []);

  const loadImage = useCallback(async (imageIndex) => {
    setLoadingImage(true);
    setError("");
    setNotice("");
    setSelected("");

    try {
      const data = await request(`/api/images/${imageIndex}`);

      setImage(`data:image/png;base64,${data.image_base64}`);
    } catch (e) {
      setImage("");
      setError(e.message);
    } finally {
      setLoadingImage(false);
    }
  }, []);

  useEffect(() => {
    loadImage(index);
    loadHistory().catch((e) => setError(e.message));
  }, [index, loadImage, loadHistory]);

  const annotatedSet = useMemo(
    () => new Set(history.map((item) => item.image_index)),
    [history]
  );

  const isAnnotated = annotatedSet.has(index);

  async function saveLabel() {
    if (!selected) return;

    setBusy(true);
    setError("");
    setNotice("");

    try {
      await request("/api/annotations", {
        method: "POST",
        body: JSON.stringify({
          image_index: index,
          chosen_label: selected,
          reveal_truth: false,
        }),
      });

      await loadHistory();

      setNotice(`Saved "${selected}" for image #${index}.`);
      setSelected("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function undoCurrent() {
    setBusy(true);
    setError("");
    setNotice("");

    try {
      await request(`/api/annotations/${index}`, {
        method: "DELETE",
      });

      await loadHistory();

      setNotice(`Annotation for image #${index} removed.`);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function move(delta) {
    setIndex((current) =>
      Math.max(0, Math.min(49999, current + delta))
    );
  }

  return (
    <motion.section
      className="studio-page"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="studio-heading">
        <div>
          <div className="studio-eyebrow">
            <Target size={14} />
            HUMAN-IN-THE-LOOP WORKFLOW
          </div>

          <h1>Annotation Studio</h1>

          <p>
            Review a CIFAR-10 image, choose a class, and
            persist your annotation.
          </p>
        </div>

        <div className="studio-count">
          <Database size={16} />
          {total.toLocaleString()} saved labels
        </div>
      </div>

      <div className="studio-layout">
        <div className="studio-image-panel">
          <div className="studio-panel-top">
            <span>TRAINING IMAGE</span>

            <span
              className={
                isAnnotated ? "state-tag done" : "state-tag"
              }
            >
              {isAnnotated ? "ANNOTATED" : "UNLABELED"}
            </span>
          </div>

          <div className="image-stage">
            {loadingImage ? (
              <div className="studio-loading">
                <LoaderCircle className="spin" />
                Loading image...
              </div>
            ) : image ? (
              <img
                src={image}
                alt={`CIFAR-10 training sample ${index}`}
              />
            ) : (
              <div className="studio-loading">
                Image unavailable
              </div>
            )}
          </div>

          <div className="image-controls">
            <button
              className="studio-button secondary"
              onClick={() => move(-1)}
              disabled={index === 0 || busy}
            >
              <ArrowLeft size={16} />
              Previous
            </button>

            <div className="image-index">
              Sample <strong>#{index}</strong>
              <span> / 49,999</span>
            </div>

            <button
              className="studio-button secondary"
              onClick={() => move(1)}
              disabled={index === 49999 || busy}
            >
              Next
              <ArrowRight size={16} />
            </button>
          </div>

          <div className="jump-row">
            <label htmlFor="jump-image">Jump to image</label>

            <input
              id="jump-image"
              type="number"
              min="0"
              max="49999"
              value={index}
              onChange={(e) =>
                setIndex(
                  Math.max(
                    0,
                    Math.min(49999, Number(e.target.value) || 0)
                  )
                )
              }
            />

            <button
              className="studio-icon-button"
              title="Reload image"
              onClick={() => loadImage(index)}
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>

        <div className="studio-label-panel">
          <div className="studio-panel-top">
            <span>SELECT A CLASS</span>
            <span className="class-hint">10 CIFAR-10 classes</span>
          </div>

          <div className="class-grid">
            {CLASSES.map((name, i) => (
              <button
                key={name}
                className={`class-option ${
                  selected === name ? "selected" : ""
                }`}
                onClick={() => setSelected(name)}
                disabled={busy}
              >
                <span className="class-number">
                  {String(i + 1).padStart(2, "0")}
                </span>

                <span>{name}</span>

                {selected === name && <Check size={16} />}
              </button>
            ))}
          </div>

          <div className="annotation-note">
            <strong>Annotation protocol</strong>

            <p>
              The image endpoint does not expose its ground-truth
              label. Your selection is saved as the annotator's
              label; ground truth remains hidden in this workflow.
            </p>
          </div>

          {error && (
            <div className="studio-alert error">{error}</div>
          )}

          {notice && (
            <div className="studio-alert success">{notice}</div>
          )}

          <button
            className="studio-button save-button"
            onClick={saveLabel}
            disabled={!selected || busy || loadingImage}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Save size={16} />
            )}
            Save annotation
          </button>

          <button
            className="studio-button undo-button"
            onClick={undoCurrent}
            disabled={!isAnnotated || busy}
          >
            <RotateCcw size={15} />
            Undo annotation for this image
          </button>
        </div>
      </div>

      <div className="history-panel">
        <div className="history-heading">
          <div>
            <h2>Recent annotations</h2>
            <p>Persisted in your local SQLite database</p>
          </div>

          <button
            className="studio-icon-button"
            onClick={() =>
              loadHistory().catch((e) => setError(e.message))
            }
            title="Refresh history"
          >
            <RefreshCw size={16} />
          </button>
        </div>

        {history.length === 0 ? (
          <div className="empty-history">
            No annotations saved yet. Select a class above to begin.
          </div>
        ) : (
          <div className="history-list">
            {history.slice(0, 8).map((item) => (
              <button
                className={`history-item ${
                  item.image_index === index ? "current" : ""
                }`}
                key={item.id}
                onClick={() => setIndex(item.image_index)}
              >
                <span className="history-id">
                  #{item.image_index}
                </span>

                <span className="history-label">
                  {item.chosen_label}
                </span>

                <span className="history-time">
                  {item.created_at}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </motion.section>
  );
}

/* =========================================================
   RESEARCH DASHBOARD
========================================================= */

function ResearchDashboard({ onOpenStudio, annotationCount }) {
  const [view, setView] = useState("validation");
  const [metrics, setMetrics] = useState(null);
  const [modelExperiments, setModelExperiments] = useState([]);
  const [metricsError, setMetricsError] = useState("");
  const [loadingMetrics, setLoadingMetrics] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [targetAccuracy, setTargetAccuracy] = useState(60);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadDashboardData = useCallback(async (manual = false) => {
    if (manual) {
      setRefreshing(true);
    } else {
      setLoadingMetrics(true);
    }

    setMetricsError("");

    try {
      const [metricsResult, modelResult] = await Promise.allSettled([
        request("/api/experiments/metrics"),
        request("/api/model-lab/experiments"),
      ]);

      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value);
      } else {
        throw metricsResult.reason;
      }

      if (modelResult.status === "fulfilled") {
        const result = modelResult.value;
        const items = Array.isArray(result)
          ? result
          : result.items || result.experiments || [];

        setModelExperiments(items);
      }

      setLastUpdated(new Date());
    } catch (error) {
      setMetricsError(
        error.message || "Unable to load experiment metrics."
      );
    } finally {
      setLoadingMetrics(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const files = metrics?.files || {};

  const aggregate = Array.isArray(
    files["multi_seed_aggregate.csv"]
  )
    ? files["multi_seed_aggregate.csv"]
    : [];

  const testSummaryRaw =
    files["multi_seed_test_summary.json"];

  const testAggregate = Array.isArray(
    files["multi_seed_test_aggregate.csv"]
  )
    ? files["multi_seed_test_aggregate.csv"]
    : [];

  const summary = files["multi_seed_summary.json"] || {};

  const curveData = useMemo(() => {
    const byBudget = new Map();

    aggregate.forEach((row) => {
      const budget = Number(row.budget);
      const mean = Number(row.mean);

      if (!Number.isFinite(budget) || !Number.isFinite(mean)) {
        return;
      }

      if (!byBudget.has(budget)) {
        byBudget.set(budget, { labels: budget });
      }

      const strategy = String(row.strategy || "").toLowerCase();

      if (strategy === "random") {
        byBudget.get(budget).random = mean;
      }

      if (strategy === "uncertainty") {
        byBudget.get(budget).uncertainty = mean;
      }
    });

    return [...byBudget.values()].sort(
      (a, b) => a.labels - b.labels
    );
  }, [aggregate]);

  const testRows = useMemo(() => {
    const source = testAggregate.length
      ? testAggregate
      : Array.isArray(testSummaryRaw)
        ? testSummaryRaw
        : Array.isArray(summary.test_results)
          ? summary.test_results
          : [];

    return source
      .map((row) => ({
        strategy:
          String(row.strategy || "").toLowerCase() ===
          "uncertainty"
            ? "Uncertainty"
            : "Random",
        accuracy: Number(row.mean),
        spread: Number(row.std),
      }))
      .filter(
        (row) =>
          Number.isFinite(row.accuracy) &&
          Number.isFinite(row.spread)
      );
  }, [testAggregate, testSummaryRaw, summary]);

  const seeds = Array.isArray(summary.seeds)
    ? summary.seeds
    : [];

  const seedCount = seeds.length;

  const strategies = Array.isArray(summary.strategies)
    ? summary.strategies.length
    : 2;

  const finalBudget = Math.max(
    ...curveData.map((row) => row.labels),
    Number(summary.max_budget) || 0,
    0
  );

  const hasCurve = curveData.some(
    (row) =>
      Number.isFinite(row.random) ||
      Number.isFinite(row.uncertainty)
  );

  const hasTest = testRows.length > 0;

  const efficiencyRows = useMemo(() => {
    const targets = [50, 60, 70, 80, 90];

    function budgetFor(strategy, target) {
      const points = curveData
        .filter((row) => Number.isFinite(row[strategy]))
        .sort((a, b) => a.labels - b.labels);

      const reached = points.find(
        (row) => row[strategy] >= target
      );

      return reached ? reached.labels : null;
    }

    return targets.map((target) => ({
      target,
      random: budgetFor("random", target),
      uncertainty: budgetFor("uncertainty", target),
    }));
  }, [curveData]);

  const selectedEfficiency = efficiencyRows.find(
    (row) => row.target === targetAccuracy
  );

  const interpretation = hasTest
    ? `In this experiment snapshot, ${testRows
        .map(
          (row) =>
            `${row.strategy.toLowerCase()} sampling has ${row.accuracy.toFixed(
              2
            )}% mean test accuracy (SD ${row.spread.toFixed(2)})`
        )
        .join("; ")}. These results are descriptive. The number of seeds and experimental setup limit how broadly they can be generalized.`
    : "Final test results are not available yet. Once the experiment summary files are present, this panel will report the observed values.";

  const recentExperiments = modelExperiments.slice(0, 5);

  return (
    <motion.section
      className="research-page"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="research-heading">
        <div>
          <div className="overview-kicker">
            <ChartNoAxesCombined size={15} />
            EXPERIMENTAL OVERVIEW
          </div>

          <h1>Research Dashboard</h1>

          <p>
            Explore active-learning results, annotation efficiency,
            and reproducibility across experimental runs.
          </p>
        </div>

        <div className="research-heading-actions">
          <span className="snapshot-badge">
            <Activity size={14} />
            {seedCount || "—"} seeds ·{" "}
            {metrics?.available ? "live data" : "checking data"}
          </span>

          <button
            className="studio-button secondary"
            onClick={() => loadDashboardData(true)}
            disabled={refreshing || loadingMetrics}
          >
            <RefreshCw
              size={15}
              className={refreshing ? "spin" : ""}
            />
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {lastUpdated && (
        <div className="dashboard-updated">
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      )}

      <div className="metric-grid">
        <article className="metric-card">
          <span>Dataset</span>
          <strong>{summary.dataset || "CIFAR-10"}</strong>
          <small>50,000 training · 10,000 test</small>
        </article>

        <article className="metric-card">
          <span>Strategies</span>
          <strong>{strategies}</strong>
          <small>Random and uncertainty sampling</small>
        </article>

        <article className="metric-card">
          <span>Final label budget</span>
          <strong>
            {finalBudget ? finalBudget.toLocaleString() : "—"}
          </strong>
          <small>From aggregated experiment data</small>
        </article>

        <article className="metric-card">
          <span>Local annotations</span>
          <strong>{annotationCount.toLocaleString()}</strong>
          <small>Saved in SQLite</small>
        </article>
      </div>

      {loadingMetrics && (
        <div className="studio-alert">
          Loading experiment artifacts...
        </div>
      )}

      {metricsError && (
        <div className="studio-alert error">
          Metrics API error: {metricsError}
        </div>
      )}

      {metrics?.missing_files?.length > 0 && (
        <div className="studio-alert error">
          Missing experiment files:{" "}
          {metrics.missing_files.join(", ")}. Charts will display
          only available data.
        </div>
      )}

      <div className="research-panel">
        <div className="research-panel-head">
          <div>
            <h2>Learning curve</h2>
            <p>
              Mean validation accuracy across labeled-data budgets
            </p>
          </div>

          <div className="chart-switch">
            <button
              className={view === "validation" ? "selected" : ""}
              onClick={() => setView("validation")}
            >
              Validation
            </button>

            <button
              className={view === "test" ? "selected" : ""}
              onClick={() => setView("test")}
            >
              Final test
            </button>
          </div>
        </div>

        {view === "validation" ? (
          hasCurve ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={curveData}
                  margin={{
                    top: 12,
                    right: 20,
                    left: 0,
                    bottom: 8,
                  }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#2b2e40"
                  />

                  <XAxis
                    dataKey="labels"
                    stroke="#858ba0"
                    tick={{ fontSize: 11 }}
                    label={{
                      value: "Labeled training samples",
                      position: "insideBottom",
                      offset: -2,
                      fill: "#858ba0",
                      fontSize: 11,
                    }}
                  />

                  <YAxis
                    domain={[0, 100]}
                    stroke="#858ba0"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value) => `${value}%`}
                  />

                  <Tooltip
                    contentStyle={{
                      background: "#171927",
                      border: "1px solid #393d52",
                      borderRadius: 8,
                      color: "#eee",
                    }}
                    formatter={(value) => [
                      `${Number(value).toFixed(2)}%`,
                      "Mean validation accuracy",
                    ]}
                  />

                  <Legend />

                  <Line
                    type="monotone"
                    dataKey="random"
                    name="Random sampling"
                    stroke="#9b8cff"
                    strokeWidth={2.5}
                    dot={{ r: 3 }}
                    connectNulls
                  />

                  <Line
                    type="monotone"
                    dataKey="uncertainty"
                    name="Uncertainty sampling"
                    stroke="#42cbd7"
                    strokeWidth={2.5}
                    dot={{ r: 3 }}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty-history">
              No aggregated validation metrics are available.
              Check experiments/multi_seed_aggregate.csv.
            </div>
          )
        ) : hasTest ? (
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={testRows}
                margin={{
                  top: 20,
                  right: 20,
                  left: 0,
                  bottom: 5,
                }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#2b2e40"
                />

                <XAxis
                  dataKey="strategy"
                  stroke="#858ba0"
                  tick={{ fontSize: 11 }}
                />

                <YAxis
                  domain={[0, 100]}
                  stroke="#858ba0"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => `${value}%`}
                />

                <Tooltip
                  contentStyle={{
                    background: "#171927",
                    border: "1px solid #393d52",
                    borderRadius: 8,
                    color: "#eee",
                  }}
                  formatter={(value, name, props) => [
                    `${Number(value).toFixed(2)}% ± ${props.payload.spread.toFixed(
                      2
                    )}`,
                    "Mean test accuracy",
                  ]}
                />

                <Bar
                  dataKey="accuracy"
                  name="Mean test accuracy"
                  fill="#9b8cff"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="empty-history">
            No final test summary is available. Check the test
            summary CSV/JSON files.
          </div>
        )}

        <div className="chart-footnote">
          Validation and test results are reported separately.
          Standard deviation describes variation across seeds.
          This is an exploratory snapshot, not a universal claim.
        </div>
      </div>

      <div className="research-bottom-grid">
        <div className="research-panel">
          <h2>Annotation efficiency</h2>
          <p>
            Minimum observed label budget required to reach a
            selected validation-accuracy target.
          </p>

          <div className="efficiency-selector">
            <label htmlFor="accuracy-target">
              Target accuracy
            </label>

            <select
              id="accuracy-target"
              value={targetAccuracy}
              onChange={(event) =>
                setTargetAccuracy(Number(event.target.value))
              }
            >
              {[50, 60, 70, 80, 90].map((target) => (
                <option key={target} value={target}>
                  {target}%
                </option>
              ))}
            </select>
          </div>

          {selectedEfficiency ? (
            <div className="efficiency-results">
              <div className="efficiency-result">
                <span>Random sampling</span>
                <strong>
                  {selectedEfficiency.random !== null
                    ? selectedEfficiency.random.toLocaleString()
                    : "Not reached"}
                </strong>
              </div>

              <div className="efficiency-result">
                <span>Uncertainty sampling</span>
                <strong>
                  {selectedEfficiency.uncertainty !== null
                    ? selectedEfficiency.uncertainty.toLocaleString()
                    : "Not reached"}
                </strong>
              </div>
            </div>
          ) : (
            <div className="empty-history">
              No efficiency data available.
            </div>
          )}

          <small className="chart-footnote">
            Values are based on observed budget points, not
            interpolated estimates.
          </small>
        </div>

        <div className="research-panel">
          <h2>Reproducibility</h2>
          <p>
            Summary of the multi-seed experiment configuration.
          </p>

          <div className="repro-grid">
            <div>
              <span>Independent seeds</span>
              <strong>{seedCount || "—"}</strong>
            </div>

            <div>
              <span>Strategies</span>
              <strong>{strategies}</strong>
            </div>

            <div>
              <span>Budgets evaluated</span>
              <strong>{curveData.length || "—"}</strong>
            </div>

            <div>
              <span>Test summaries</span>
              <strong>{testRows.length || "—"}</strong>
            </div>
          </div>

          <div className="seed-list">
            <span>Seed values</span>
            <p>
              {seedCount ? seeds.join(", ") : "Not available"}
            </p>
          </div>
        </div>
      </div>

      <div className="research-bottom-grid">
        <div className="research-panel">
          <h2>Final test summary</h2>

          <div className="result-table">
            <div className="result-row header">
              <span>Strategy</span>
              <span>Mean ± SD</span>
            </div>

            {testRows.map((item) => (
              <div className="result-row" key={item.strategy}>
                <span>{item.strategy}</span>
                <strong>
                  {item.accuracy.toFixed(2)}% ±{" "}
                  {item.spread.toFixed(2)}
                </strong>
              </div>
            ))}
          </div>
        </div>

        <div className="research-panel research-note">
          <h2>Research interpretation</h2>
          <p>{interpretation}</p>

          <button
            className="studio-button"
            onClick={onOpenStudio}
          >
            Open Annotation Studio
            <ArrowRight size={15} />
          </button>
        </div>
      </div>

      <div className="research-panel model-experiments-panel">
        <div className="research-panel-head">
          <div>
            <h2>Recent Model Lab experiments</h2>
            <p>
              Exploratory training runs, shown separately from
              multi-seed research results.
            </p>
          </div>
        </div>

        {recentExperiments.length ? (
          <div className="result-table">
            <div className="result-row model-experiment-row header">
              <span>Experiment</span>
              <span>Epochs</span>
              <span>Best validation</span>
              <span>Test accuracy</span>
              <span>Status</span>
            </div>

            {recentExperiments.map((experiment) => {
              const evaluation = experiment.evaluation || {};

              const testAccuracy =
                evaluation.test_accuracy ??
                experiment.test_accuracy;

              const validationAccuracy =
                experiment.best_val_accuracy ??
                experiment.best_validation_accuracy;

              return (
                <div
                  className="result-row model-experiment-row"
                  key={experiment.experiment_id || experiment.id}
                >
                  <span
                    title={
                      experiment.experiment_id || experiment.id
                    }
                  >
                    {(
                      experiment.experiment_id ||
                      experiment.id ||
                      "Run"
                    )
                      .toString()
                      .slice(0, 22)}
                  </span>

                  <span>{experiment.epochs ?? "—"}</span>

                  <strong>
                    {Number.isFinite(Number(validationAccuracy))
                      ? `${Number(validationAccuracy).toFixed(2)}%`
                      : "—"}
                  </strong>

                  <strong>
                    {Number.isFinite(Number(testAccuracy))
                      ? `${Number(testAccuracy).toFixed(2)}%`
                      : "—"}
                  </strong>

                  <span>
                    {experiment.status || "Completed"}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty-history">
            {loadingMetrics
              ? "Loading Model Lab history..."
              : "No Model Lab experiments found yet."}
          </div>
        )}

        <div className="chart-footnote">
          Model Lab results may use different training configurations
          and label budgets. Do not compare them directly with the
          multi-seed study without matching experimental conditions.
        </div>
      </div>
    </motion.section>
  );
}

/* =========================================================
   DATASET EXPLORER
========================================================= */

function DatasetExplorer({ onOpenStudio }) {
  const [summary, setSummary] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [images, setImages] = useState([]);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(24);
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const annotationMap = useMemo(
    () =>
      new Map(
        annotations.map((item) => [
          Number(item.image_index),
          item,
        ])
      ),
    [annotations]
  );

  const loadExplorerData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [summaryData, annotationData] = await Promise.all([
        request("/api/dataset/summary"),
        request("/api/annotations"),
      ]);

      setSummary(summaryData);
      setAnnotations(annotationData.items || []);
    } catch (e) {
      setError(
        e.message || "Could not load dataset information."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadExplorerData();
  }, [loadExplorerData]);

  const totalImages = Number(
    summary?.total_training_images ??
      summary?.train_size ??
      summary?.training_images ??
      summary?.total_images ??
      50000
  );

  const annotatedCount = annotations.length;

  const filteredIndices = useMemo(() => {
    const query = search.trim();
    const result = [];

    for (let i = 0; i < totalImages; i += 1) {
      const annotated = annotationMap.has(i);

      if (filter === "annotated" && !annotated) continue;
      if (filter === "unlabeled" && annotated) continue;
      if (query && !String(i).includes(query)) continue;

      result.push(i);
    }

    return result;
  }, [totalImages, annotationMap, filter, search]);

  const pageCount = Math.max(
    1,
    Math.ceil(filteredIndices.length / pageSize)
  );

  const visibleIndices = filteredIndices.slice(
    page * pageSize,
    (page + 1) * pageSize
  );

  useEffect(() => {
    let cancelled = false;

    async function loadThumbnails() {
      setImages([]);

      if (!visibleIndices.length) return;

      const results = await Promise.all(
        visibleIndices.map(async (index) => {
          try {
            const data = await request(`/api/images/${index}`);

            return {
              index,
              src: `data:image/png;base64,${data.image_base64}`,
              error: "",
            };
          } catch (e) {
            return {
              index,
              src: "",
              error: e.message,
            };
          }
        })
      );

      if (!cancelled) {
        setImages(results);
      }
    }

    loadThumbnails();

    return () => {
      cancelled = true;
    };
  }, [visibleIndices.join(",")]);

  const openImage = async (index) => {
    setSelectedImage({
      index,
      src: "",
      loading: true,
      error: "",
    });

    try {
      const data = await request(`/api/images/${index}`);

      setSelectedImage({
        index,
        src: `data:image/png;base64,${data.image_base64}`,
        loading: false,
        error: "",
      });
    } catch (e) {
      setSelectedImage({
        index,
        src: "",
        loading: false,
        error: e.message,
      });
    }
  };

  const classCounts = useMemo(() => {
    const counts = Object.fromEntries(
      CLASSES.map((name) => [name, 0])
    );

    annotations.forEach((item) => {
      if (Object.hasOwn(counts, item.chosen_label)) {
        counts[item.chosen_label] += 1;
      }
    });

    return counts;
  }, [annotations]);

  return (
    <motion.section
      className="research-page dataset-page"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="research-heading">
        <div>
          <div className="overview-kicker">
            <Images size={15} />
            DATASET INSPECTION
          </div>

          <h1>Dataset Explorer</h1>

          <p>
            Browse CIFAR-10 training samples and inspect your
            annotation coverage.
          </p>
        </div>

        <button
          className="studio-button secondary"
          onClick={loadExplorerData}
          disabled={loading}
        >
          <RefreshCw size={15} />
          Refresh data
        </button>
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <span>Training images</span>
          <strong>{totalImages.toLocaleString()}</strong>
          <small>CIFAR-10 training split</small>
        </article>

        <article className="metric-card">
          <span>Annotated by you</span>
          <strong>{annotatedCount.toLocaleString()}</strong>
          <small>Persisted in local SQLite</small>
        </article>

        <article className="metric-card">
          <span>Remaining unlabeled</span>
          <strong>
            {Math.max(
              0,
              totalImages - annotatedCount
            ).toLocaleString()}
          </strong>
          <small>Based on saved annotations</small>
        </article>

        <article className="metric-card">
          <span>Annotation coverage</span>
          <strong>
            {totalImages
              ? `${(
                  (annotatedCount / totalImages) *
                  100
                ).toFixed(2)}%`
              : "—"}
          </strong>
          <small>Of training images</small>
        </article>
      </div>

      <div className="research-panel">
        <div className="research-panel-head">
          <div>
            <h2>Training image gallery</h2>
            <p>
              Image content is retrieved from the API; hidden
              ground-truth labels are never requested.
            </p>
          </div>

          <button
            className="studio-button"
            onClick={onOpenStudio}
          >
            <Target size={15} />
            Open Annotation Studio
          </button>
        </div>

        <div className="dataset-toolbar">
          <div className="dataset-search">
            <Search size={16} />

            <input
              aria-label="Search image index"
              placeholder="Search by image index..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
            />
          </div>

          <div className="dataset-filter">
            <Filter size={16} />

            <select
              aria-label="Filter images"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(0);
              }}
            >
              <option value="all">All images</option>
              <option value="annotated">Annotated</option>
              <option value="unlabeled">Unlabeled</option>
            </select>
          </div>

          <select
            aria-label="Images per page"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(0);
            }}
          >
            <option value={12}>12 / page</option>
            <option value={24}>24 / page</option>
            <option value={48}>48 / page</option>
          </select>
        </div>

        {error && (
          <div className="studio-alert error">{error}</div>
        )}

        {loading ? (
          <div className="empty-history">
            Loading dataset...
          </div>
        ) : visibleIndices.length === 0 ? (
          <div className="empty-history">
            No images match this filter.
          </div>
        ) : (
          <div className="dataset-grid">
            {visibleIndices.map((index) => {
              const item = images.find(
                (image) => image.index === index
              );

              const annotation = annotationMap.get(index);

              return (
                <button
                  className="dataset-tile"
                  key={index}
                  onClick={() => openImage(index)}
                  title={`Inspect image #${index}`}
                >
                  <div className="dataset-thumb">
                    {item?.src ? (
                      <img
                        src={item.src}
                        alt={`CIFAR-10 sample ${index}`}
                      />
                    ) : (
                      <span>
                        {item?.error
                          ? "Unavailable"
                          : "Loading..."}
                      </span>
                    )}
                  </div>

                  <div className="dataset-tile-meta">
                    <strong>#{index}</strong>

                    <span
                      className={
                        annotation
                          ? "state-tag done"
                          : "state-tag"
                      }
                    >
                      {annotation ? "Annotated" : "Unlabeled"}
                    </span>
                  </div>

                  {annotation && (
                    <small className="dataset-label">
                      {annotation.chosen_label}
                    </small>
                  )}
                </button>
              );
            })}
          </div>
        )}

        <div className="dataset-pagination">
          <span>
            Showing{" "}
            {filteredIndices.length
              ? page * pageSize + 1
              : 0}
            –
            {Math.min(
              (page + 1) * pageSize,
              filteredIndices.length
            )}{" "}
            of {filteredIndices.length.toLocaleString()}
          </span>

          <div>
            <button
              className="studio-button secondary"
              onClick={() =>
                setPage((p) => Math.max(0, p - 1))
              }
              disabled={page === 0}
            >
              Previous
            </button>

            <span>
              Page {page + 1} / {pageCount}
            </span>

            <button
              className="studio-button secondary"
              onClick={() =>
                setPage((p) =>
                  Math.min(pageCount - 1, p + 1)
                )
              }
              disabled={page >= pageCount - 1}
            >
              Next
            </button>
          </div>
        </div>
      </div>

      <div className="research-panel">
        <div className="research-panel-head">
          <div>
            <h2>Your annotation distribution</h2>
            <p>
              Counts reflect only labels you have saved, not
              the hidden CIFAR-10 targets.
            </p>
          </div>
        </div>

        <div className="dataset-class-list">
          {CLASSES.map((name) => (
            <div className="dataset-class-row" key={name}>
              <span>{name}</span>

              <div className="dataset-class-track">
                <div
                  style={{
                    width: `${
                      annotatedCount
                        ? (classCounts[name] / annotatedCount) *
                          100
                        : 0
                    }%`,
                  }}
                />
              </div>

              <strong>{classCounts[name]}</strong>
            </div>
          ))}
        </div>
      </div>

      {selectedImage && (
        <div
          className="dataset-modal-backdrop"
          role="presentation"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="dataset-modal"
            role="dialog"
            aria-modal="true"
            aria-label={`Image ${selectedImage.index} preview`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="dataset-modal-head">
              <div>
                <strong>
                  Training sample #{selectedImage.index}
                </strong>

                <p>
                  {annotationMap.has(selectedImage.index)
                    ? `Your label: ${
                        annotationMap.get(
                          selectedImage.index
                        ).chosen_label
                      }`
                    : "No saved annotation"}
                </p>
              </div>

              <button
                className="studio-icon-button"
                onClick={() => setSelectedImage(null)}
                aria-label="Close preview"
              >
                ×
              </button>
            </div>

            <div className="dataset-modal-image">
              {selectedImage.loading
                ? "Loading image..."
                : selectedImage.src
                  ? (
                    <img
                      src={selectedImage.src}
                      alt={`CIFAR-10 sample ${selectedImage.index} enlarged`}
                    />
                  )
                  : selectedImage.error || "Image unavailable"}
            </div>

            <div className="dataset-modal-actions">
              <button
                className="studio-button"
                onClick={() => {
                  setSelectedImage(null);
                  onOpenStudio();
                }}
              >
                Annotate in Studio
                <ArrowRight size={15} />
              </button>

              <button
                className="studio-button secondary"
                onClick={() => setSelectedImage(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </motion.section>
  );
}

/* =========================================================
   MAIN APPLICATION
========================================================= */

export default function App() {
  const [page, setPage] = useState("Overview");
  const [apiStatus, setApiStatus] = useState("checking");
  const [annotationCount, setAnnotationCount] = useState(0);

  const refreshAnnotationCount = useCallback(async () => {
    try {
      const data = await request("/api/annotations");
      setAnnotationCount(data.total || 0);
    } catch {
      // Keep the last known count if the API is unavailable.
    }
  }, []);

  useEffect(() => {
    request("/api/health")
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));

    refreshAnnotationCount();
  }, [refreshAnnotationCount]);

  return (
    <div className="oracle-shell">
      <aside className="oracle-sidebar">
        <div className="oracle-brand">
          <div className="oracle-mark">◈</div>

          <div>
            <strong>
              ORACLE<span>.</span>
            </strong>

            <small>RESEARCH INTELLIGENCE</small>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <button
          className={`oracle-nav ${
            page === "Overview" ? "active" : ""
          }`}
          onClick={() => setPage("Overview")}
        >
          <LayoutDashboard size={18} />
          Overview
        </button>

        <button
          className={`oracle-nav ${
            page === "Research Dashboard" ? "active" : ""
          }`}
          onClick={() => setPage("Research Dashboard")}
        >
          <FlaskConical size={18} />
          Research Dashboard
        </button>

        <button
          className={`oracle-nav ${
            page === "Annotation Studio" ? "active" : ""
          }`}
          onClick={() => setPage("Annotation Studio")}
        >
          <Target size={18} />
          Annotation Studio
        </button>

        <button
          className={`oracle-nav ${
            page === "Dataset Explorer" ? "active" : ""
          }`}
          onClick={() => setPage("Dataset Explorer")}
        >
          <Images size={18} />
          Dataset Explorer
        </button>

        <button
          className={`oracle-nav ${
            page === "Model Lab" ? "active" : ""
          }`}
          onClick={() => setPage("Model Lab")}
        >
          <Activity size={18} />
          Model Lab
        </button>

        <div className="sidebar-status">
          <span
            className={
              apiStatus === "online"
                ? "status-dot"
                : "status-dot offline"
            }
          />

          {apiStatus === "online"
            ? "API connected"
            : apiStatus === "checking"
              ? "Connecting to API..."
              : "API offline"}
        </div>

        <div className="sidebar-foot">
          Open Research for Active Learning and Continuous Evaluation
        </div>
      </aside>

      <main className="oracle-main">
        <header className="oracle-topbar">
          <span>
            Workspace / <strong>{page}</strong>
          </span>

          <span className="local-badge">
            LOCAL RESEARCH ENVIRONMENT
          </span>
        </header>

        {page === "Annotation Studio" ? (
          <AnnotationStudio />
        ) : page === "Dataset Explorer" ? (
          <DatasetExplorer
            onOpenStudio={() => {
              setPage("Annotation Studio");
              refreshAnnotationCount();
            }}
          />
        ) : page === "Research Dashboard" ? (
          <ResearchDashboard
            onOpenStudio={() => {
              setPage("Annotation Studio");
              refreshAnnotationCount();
            }}
            annotationCount={annotationCount}
          />
        ) : page === "Model Lab" ? (
          <ModelLab />
        ) : (
          <section className="overview-page">
            <div className="overview-kicker">
              <Activity size={15} />
              ACTIVE LEARNING RESEARCH
            </div>

            <h1>Good to see you, Kishan ✳</h1>

            <p className="overview-lead">
              Can informative sample selection help models learn
              with fewer labels?
            </p>

            <div className="overview-cards">
              <article>
                <Target />
                <span>Research dataset</span>
                <strong>CIFAR-10</strong>
                <small>
                  50,000 training images · 10 classes
                </small>
              </article>

              <article>
                <Database />
                <span>Annotation records</span>
                <strong>Stored locally</strong>
                <small>SQLite-backed persistence</small>
              </article>

              <article>
                <Activity />
                <span>Backend status</span>
                <strong>
                  {apiStatus === "online"
                    ? "Connected"
                    : apiStatus === "checking"
                      ? "Checking..."
                      : "Offline"}
                </strong>
                <small>FastAPI · localhost:8000</small>
              </article>
            </div>

            <div className="overview-cta">
              <h2>Ready to label your first image?</h2>

              <p>
                Open Annotation Studio to browse real CIFAR-10
                samples and save labels to your local database.
              </p>

              <button
                onClick={() => setPage("Annotation Studio")}
              >
                Open Annotation Studio
                <ArrowRight size={16} />
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}