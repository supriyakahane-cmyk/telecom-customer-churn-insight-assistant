from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from churn_assistant import (
    SummaryGenerationError,
    analyze_customer,
    resolve_supervisor_review,
)


HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 1_000_000

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f7f8fa">
  <title>Telecom Churn Insight Assistant</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #182230;
      background: #f7f8fa;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
      --border: #e4e7ec;
      --muted: #667085;
      --blue: #2475e8;
      --green: #14804a;
      --amber: #a15c07;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    .shell { max-width: 1220px; margin: 36px auto; padding: 0 24px 36px; }
    .page-heading { margin: 0 0 7px; font-size: 26px; font-weight: 700; letter-spacing: -.04em; }
    .page-subheading { margin: 0 0 24px; color: var(--muted); font-size: 14px; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, .94fr) minmax(0, 1.06fr);
      background: #fff;
      border: 1px solid #d9dde4;
      border-radius: 13px;
      box-shadow: 0 8px 28px rgb(16 24 40 / 5%);
      overflow: hidden;
    }
    .panel { min-width: 0; padding: 23px 26px 25px; }
    .request-panel { border-right: 1px solid var(--border); }
    .panel-title { margin: 0 0 20px; font-size: 16px; font-weight: 680; letter-spacing: -.015em; }
    .field { margin-bottom: 15px; }
    label { display: block; margin: 0 0 6px; color: #344054; font-size: 12px; font-weight: 620; }
    input, textarea, select {
      display: block;
      width: 100%;
      min-height: 42px;
      padding: 10px 12px;
      color: #182230;
      background: #fff;
      border: 1px solid #d0d5dd;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 4%);
      font: inherit;
      font-size: 13px;
      transition: border-color .15s, box-shadow .15s;
    }
    textarea { min-height: 75px; resize: vertical; }
    input:focus, textarea:focus, select:focus { outline: 0; border-color: var(--blue); box-shadow: 0 0 0 3px rgb(36 117 232 / 13%); }
    .primary-button, .review-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 41px;
      padding: 9px 16px;
      border: 1px solid transparent;
      border-radius: 8px;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
      transition: background .15s, border-color .15s, transform .15s;
    }
    .primary-button { min-width: 142px; margin-top: 2px; color: #fff; background: var(--blue); box-shadow: 0 1px 2px rgb(16 24 40 / 12%); }
    .primary-button:hover { background: #165fc7; }
    .primary-button:disabled, .review-button:disabled { opacity: .65; cursor: wait; }
    .privacy-note { margin: 18px 0 0; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .response-panel { padding-left: 25px; }
    .placeholder { display: grid; min-height: 270px; place-items: center; padding: 30px; color: var(--muted); text-align: center; font-size: 13px; line-height: 1.6; }
    .placeholder-icon { display: grid; width: 43px; height: 43px; margin: 0 auto 12px; place-items: center; border-radius: 12px; color: #175cd3; background: #eff6ff; font-size: 20px; }
    .hidden { display: none !important; }
    .metrics { display: grid; grid-template-columns: .82fr 1.18fr; gap: 12px; margin-bottom: 19px; }
    .metric-card { min-width: 0; padding: 12px 14px; border: 1px solid var(--border); border-radius: 9px; background: #fff; }
    .metric-label { display: block; margin-bottom: 7px; color: var(--muted); font-size: 11px; font-weight: 560; }
    .risk-value { display: inline-flex; align-items: center; gap: 7px; font-size: 17px; font-weight: 730; letter-spacing: -.025em; }
    .risk-value::before, .status-dot { width: 8px; height: 8px; flex: 0 0 8px; border-radius: 50%; background: #98a2b3; }
    .risk-high .risk-value { color: #b42318; }
    .risk-high .risk-value::before { background: #e04448; }
    .risk-high { border-color: #fda29b; background: #fef3f2; }
    .risk-medium .risk-value { color: #9a6700; }
    .risk-medium .risk-value::before { background: #eaaa08; }
    .risk-medium { border-color: #fedf89; background: #fffaeb; }
    .risk-low .risk-value { color: #067647; }
    .risk-low .risk-value::before { background: #17a56b; }
    .risk-low { border-color: #abefc6; background: #ecfdf3; }
    .risk-uncertain .risk-value { color: #475467; font-size: 14px; }
    .score-caption { margin-top: 3px; color: var(--muted); font-size: 11px; }
    .action-card { border-color: #73c995; background: #f0fbf4; }
    .action-card .metric-label { color: #33744c; }
    .action-name { font-size: 14px; font-weight: 700; }
    .action-reason { margin-top: 4px; color: #475467; font-size: 11px; line-height: 1.45; }
    .no-action { border-color: #e4c887; background: #fffaeb; }
    .summary { margin: 0 0 18px; padding: 12px 14px; color: #344054; background: #f8fafc; border: 1px solid #eaecf0; border-radius: 8px; font-size: 12px; line-height: 1.55; }
    .drivers-heading { margin: 0 0 8px; font-size: 12px; font-weight: 650; }
    .driver-table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
    .driver-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 11px; }
    .driver-table th { padding: 8px 10px; color: #475467; background: #f2f4f7; font-weight: 650; white-space: nowrap; }
    .driver-table td { padding: 8px 10px; border-top: 1px solid #eaecf0; vertical-align: top; }
    .driver-table td:first-child { font-weight: 620; white-space: nowrap; }
    .source-list { color: #475467; }
    .signal-summary { margin: 9px 0 15px; color: #667085; font-size: 10px; line-height: 1.5; }
    .notice { margin: 8px 0; padding: 9px 11px; border: 1px solid; border-radius: 8px; font-size: 11px; line-height: 1.5; }
    .notice-warning { color: #805b08; background: #fffaeb; border-color: #fedf89; }
    .notice-neutral { color: #475467; background: #f2f4f7; border-color: #e4e7ec; }
    .notice-danger { color: #912018; background: #fef3f2; border-color: #fecdca; }
    .status-line { display: flex; align-items: flex-start; gap: 8px; margin-top: 13px; color: #475467; font-size: 11px; line-height: 1.5; }
    .status-line .status-dot { margin-top: 3px; background: #12a150; }
    .status-line.status-pending { color: #9a3412; }
    .status-line.status-pending .status-dot { background: #f79009; }
    .review-actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
    .review-button { min-height: 35px; padding: 7px 10px; font-size: 11px; }
    .approve-button { color: #067647; background: #ecfdf3; border-color: #abefc6; }
    .reject-button { color: #b42318; background: #fff; border-color: #fecdca; }
    .structured { margin-top: 14px; border-top: 1px solid var(--border); padding-top: 12px; }
    .structured summary { color: #475467; font-size: 11px; font-weight: 620; cursor: pointer; }
    .structured pre { max-height: 290px; overflow: auto; margin: 10px 0 0; padding: 12px; color: #344054; background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; font: 10px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .footer-note { margin: 14px 2px 0; color: var(--muted); font-size: 11px; line-height: 1.5; }
    @media (max-width: 760px) {
      .shell { margin: 24px auto; padding-right: 15px; padding-left: 15px; }
      .workspace { grid-template-columns: 1fr; }
      .request-panel { border-right: 0; border-bottom: 1px solid var(--border); }
      .panel, .response-panel { padding: 20px; }
      .response-panel { min-height: 340px; }
    }
    @media (max-width: 420px) {
      .metrics { grid-template-columns: 1fr; }
      .page-heading { font-size: 23px; }
      .panel { padding: 17px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <h1 class="page-heading">Customer insight</h1>
    <p class="page-subheading">Review verified signals and a policy-checked next step while you speak with the customer.</p>
    <main class="workspace">
      <section class="panel request-panel">
        <h2 class="panel-title">Telecom Churn Insight Assistant</h2>
        <form id="request">
          <div class="field">
            <label for="customer_id">Customer ID</label>
            <input id="customer_id" value="CUST-1001" autocomplete="off" required>
          </div>
          <div class="field">
            <label for="question">Agent question</label>
            <input id="question" value="Why might this customer leave, and what should I do?">
          </div>
          <div class="field">
            <label for="channel">Channel</label>
            <select id="channel">
              <option value="inbound_call">Inbound call</option>
              <option value="outbound_call">Outbound call</option>
              <option value="chat">Chat</option>
              <option value="email">Email</option>
            </select>
          </div>
          <div class="field">
            <label for="reason">Customer's stated reason</label>
            <textarea id="reason">Calls keep dropping at home</textarea>
          </div>
          <div class="field">
            <label for="session_date">Session date</label>
            <input id="session_date" type="date" value="2026-10-15" required>
          </div>
          <button class="primary-button" id="analyse" type="submit">Analyse customer</button>
        </form>
        <p class="privacy-note">Fictional sample records only. Do not enter names, phone numbers, addresses, account numbers, or payment-card data.</p>
      </section>
      <section class="panel response-panel" aria-live="polite" aria-busy="false">
        <h2 class="panel-title">Cited insight</h2>
        <div id="placeholder" class="placeholder">
          <div><span class="placeholder-icon" aria-hidden="true">↗</span>Enter a customer request to see verified drivers, fixed-rule risk, and one recommended next step.</div>
        </div>
        <div id="analysis" class="hidden">
          <div class="metrics">
            <div id="risk-card" class="metric-card risk-uncertain">
              <span class="metric-label">Churn risk</span>
              <div id="risk-value" class="risk-value">Not classified</div>
              <div id="risk-score" class="score-caption">Insufficient verified history</div>
            </div>
            <div id="action-card" class="metric-card action-card">
              <span class="metric-label">Recommended action</span>
              <div id="action-name" class="action-name">No action available</div>
              <div id="action-reason" class="action-reason"></div>
            </div>
          </div>
          <p id="summary" class="summary"></p>
          <h3 class="drivers-heading">Dissatisfaction drivers</h3>
          <div id="drivers"></div>
          <p id="signals" class="signal-summary"></p>
          <div id="notices"></div>
          <div id="supervisor"></div>
          <details class="structured">
            <summary>View structured response</summary>
            <pre id="raw-result"></pre>
          </details>
        </div>
        <div id="error" class="notice notice-danger hidden" role="alert"></div>
      </section>
    </main>
    <p class="footer-note">Risk is calculated from verified records in code. Every driver and recommendation links to its evidence; uncertain history and policy exceptions are sent for supervisor review.</p>
  </div>
  <script>
    const form = document.querySelector("#request");
    const analyseButton = document.querySelector("#analyse");
    const responsePanel = document.querySelector(".response-panel");
    const placeholder = document.querySelector("#placeholder");
    const analysis = document.querySelector("#analysis");
    const errorBox = document.querySelector("#error");
    const rawResult = document.querySelector("#raw-result");
    function textElement(tag, text, className) {
      const element = document.createElement(tag);
      element.textContent = text;
      if (className) element.className = className;
      return element;
    }

    function displayLabel(value) {
      return value
        .replaceAll("_", " ")
        .split(" ")
        .map(word => word ? word[0].toUpperCase() + word.slice(1) : "")
        .join(" ");
    }

    function showNotice(container, message, kind) {
      container.append(textElement("div", message, `notice notice-${kind}`));
    }

    function renderDrivers(drivers) {
      const container = document.querySelector("#drivers");
      container.replaceChildren();
      if (!drivers.length) {
        container.append(textElement("div", "No verified dissatisfaction drivers found.", "notice notice-neutral"));
        return;
      }
      const table = document.createElement("table");
      table.className = "driver-table";
      const head = document.createElement("thead");
      const headingRow = document.createElement("tr");
      for (const heading of ["Driver", "Evidence", "Source"]) {
        headingRow.append(textElement("th", heading));
      }
      head.append(headingRow);
      table.append(head);
      const body = document.createElement("tbody");
      for (const driver of drivers) {
        const row = document.createElement("tr");
        row.append(textElement("td", displayLabel(driver.driver)));
        row.append(textElement("td", driver.evidence));
        row.append(textElement("td", driver.citations.join(", "), "source-list"));
        body.append(row);
      }
      table.append(body);
      const wrap = document.createElement("div");
      wrap.className = "driver-table-wrap";
      wrap.append(table);
      container.append(wrap);
    }

    function renderResult(result) {
      placeholder.classList.add("hidden");
      errorBox.classList.add("hidden");
      analysis.classList.remove("hidden");
      document.querySelector("#summary").textContent = result.summary;
      rawResult.textContent = JSON.stringify(result, null, 2);

      const riskCard = document.querySelector("#risk-card");
      riskCard.className = `metric-card risk-${result.risk.level || "uncertain"}`;
      document.querySelector("#risk-value").textContent = result.risk.level
        ? result.risk.level[0].toUpperCase() + result.risk.level.slice(1)
        : "Not classified";
      document.querySelector("#risk-score").textContent = result.risk.score === null
        ? "Insufficient verified history"
        : `Score ${result.risk.score} · fixed scoring rule`;

      const action = result.recommended_action;
      const actionCard = document.querySelector("#action-card");
      actionCard.className = `metric-card${action ? " action-card" : " no-action"}`;
      document.querySelector("#action-name").textContent = action
        ? displayLabel(action.action)
        : "Supervisor review";
      document.querySelector("#action-reason").textContent = action
        ? `${action.reasoning} Sources: ${action.citations.join(", ")}`
        : "No grounded action is available from the verified records.";

      renderDrivers(result.drivers);
      const signalLabels = {
        network_complaints_3plus: "3+ network complaints",
        open_billing_issue: "open billing issue",
        competitor_intent: "competitor intent",
        usage_drop_30pct: "usage drop over 30%",
        repeat_contact: "repeat contact"
      };
      document.querySelector("#signals").textContent = `Score signals: ${
        result.risk.signals_used.length
          ? result.risk.signals_used.map(signal => signalLabels[signal] || signal).join(" · ")
          : "none"
      }`;

      const notices = document.querySelector("#notices");
      notices.replaceChildren();
      for (const conflict of result.data_conflicts || []) {
        showNotice(
          notices,
          `Data conflict — ${conflict.issue}: ${conflict.records.join(", ")}. ${conflict.resolution} Selected: ${conflict.selected_record}.`,
          "warning"
        );
      }
      for (const missing of result.missing_information || []) {
        showNotice(notices, `Missing: ${missing}`, "warning");
      }
      for (const unsupported of result.unsupported || []) {
        showNotice(notices, `Not supported: ${unsupported}`, "neutral");
      }

      const supervisor = document.querySelector("#supervisor");
      supervisor.replaceChildren();
      const status = result.supervisor_status || "not_required";
      const statusText = status === "pending"
        ? `Supervisor review required: ${result.escalation_reason}`
        : status === "approved"
          ? "Supervisor review completed."
          : status === "rejected"
            ? "Supervisor rejected this recommendation; no action is authorized."
            : result.escalate
              ? `Supervisor review required: ${result.escalation_reason}`
              : "Supervisor review: not needed";
      const statusLine = document.createElement("div");
      statusLine.className = `status-line${status === "pending" ? " status-pending" : ""}`;
      statusLine.append(textElement("span", "", "status-dot"));
      statusLine.append(textElement("span", statusText));
      supervisor.append(statusLine);

      if (status === "pending" && result.review_id) {
        const actions = document.createElement("div");
        actions.className = "review-actions";
        for (const [approved, label, style] of [
          [true, "Approve guarded response", "approve-button"],
          [false, "Reject recommendation", "reject-button"]
        ]) {
          const button = textElement("button", label, `review-button ${style}`);
          button.type = "button";
          button.addEventListener("click", () => resolveReview(result, approved, button));
          actions.append(button);
        }
        supervisor.append(actions);
      }
    }

    async function resolveReview(result, approved, button) {
      button.disabled = true;
      try {
        const response = await fetch("/review", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({review_id: result.review_id, approved})
        });
        const reviewed = await response.json();
        if (!response.ok) throw new Error(reviewed.error || `Request failed (${response.status})`);
        renderResult(reviewed);
      } catch (error) {
        button.disabled = false;
        errorBox.textContent = `Supervisor review failed: ${error}`;
        errorBox.classList.remove("hidden");
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      analyseButton.disabled = true;
      analyseButton.textContent = "Analysing…";
      responsePanel.setAttribute("aria-busy", "true");
      errorBox.classList.add("hidden");
      const payload = {
        customer_id: document.querySelector("#customer_id").value,
        question: document.querySelector("#question").value,
        session_date: document.querySelector("#session_date").value,
        context: {
          channel: document.querySelector("#channel").value,
          stated_reason: document.querySelector("#reason").value
        }
      };
      try {
        const response = await fetch("/insight", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
        renderResult(result);
      } catch (error) {
        errorBox.textContent = `Analysis failed: ${error}`;
        errorBox.classList.remove("hidden");
      } finally {
        analyseButton.disabled = false;
        analyseButton.textContent = "Analyse customer";
        responsePanel.setAttribute("aria-busy", "false");
      }
    });
  </script>
</body>
</html>
"""


class InsightHandler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/":
            encoded = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
        elif self.path == "/health":
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path not in {"/insight", "/review"}:
            self._write_json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._write_json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(400, {"error": "Content-Length must be a valid integer."})
            return
        if content_length <= 0:
            self._write_json(400, {"error": "A JSON request body is required."})
            return
        if content_length > MAX_BODY_BYTES:
            self._write_json(413, {"error": "Request body exceeds the 1 MB limit."})
            return

        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._write_json(400, {"error": f"Invalid JSON request: {error}"})
            return
        if not isinstance(body, dict):
            self._write_json(400, {"error": "Request body must be a JSON object."})
            return

        try:
            if self.path == "/insight":
                result = analyze_customer(
                    body, str(body.get("session_date", "2026-10-15"))
                )
            else:
                review_id = body.get("review_id")
                approved = body.get("approved")
                if not isinstance(review_id, str) or not isinstance(approved, bool):
                    self._write_json(
                        400,
                        {"error": "review_id (string) and approved (boolean) are required."},
                    )
                    return
                result = resolve_supervisor_review(review_id, approved)
        except SummaryGenerationError as error:
            self._write_json(503, {"error": str(error)})
            return
        except (TypeError, ValueError, KeyError) as error:
            self._write_json(400, {"error": str(error)})
            return
        self._write_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), InsightHandler)
    print(f"Telecom Churn Insight Assistant running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
