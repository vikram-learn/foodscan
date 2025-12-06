import React, { useState } from "react";
import { postScanImage } from "../utils/api";

export default function ScanUploader() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleUpload() {
    if (!file) {
      setError("Choose a file first.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await postScanImage(file);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err?.response?.data || err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 760, margin: "24px auto", padding: 16 }}>
      <h2 style={{ marginBottom: 12 }}>FoodScan-X — Quick Scan</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        style={{ display: "block", marginBottom: 12 }}
      />

      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            padding: "8px 14px",
            background: "#0b75ff",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "wait" : "pointer"
          }}
        >
          {loading ? "Scanning…" : "Scan Image"}
        </button>

        <button
          onClick={() => { setFile(null); setResult(null); setError(null); }}
          style={{
            padding: "8px 12px",
            background: "#eee",
            border: "1px solid #ddd",
            borderRadius: 6
          }}
        >
          Reset
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 12, color: "crimson" }}>{String(error)}</div>
      )}

      {result && (
        <div style={{ marginTop: 16, background: "#f7f7f8", padding: 12, borderRadius: 6 }}>
          <h3 style={{ marginTop: 0 }}>Result</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
