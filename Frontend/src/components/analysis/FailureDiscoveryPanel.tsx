import { useEffect, useMemo, useState } from "react";
import { AlertCircle, BrainCircuit, RefreshCw } from "lucide-react";

import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type MetadataRow = Record<string, string | number | boolean | null | undefined>;

interface Finding {
  id: string;
  rules: string[];
  sample_count: number;
  error_rate: number;
  validation_error_rate: number;
  baseline_error_rate: number;
  error_lift: number | null;
  example_ids: string[];
}

interface DiscoveryResult {
  task: "classification" | "transcription";
  record_count: number;
  baseline_error_rate: number;
  validation: {
    metric: string;
    value: number;
    sample_count: number;
  };
  feature_importance: Array<{ feature: string; importance: number }>;
  findings: Finding[];
  message?: string | null;
}

interface FailureDiscoveryPanelProps {
  model?: string;
  dataset?: string;
  predictionMap: Record<string, string>;
}

const ID_KEYS = ["id", "path", "filepath", "file", "filename"];
const TRANSCRIPT_KEYS = ["sentence", "transcript", "text", "statement"];
const LABEL_KEYS = ["emotion", "label", "class", "target"];
const EXCLUDED_FEATURE_KEYS = new Set([
  ...ID_KEYS,
  ...TRANSCRIPT_KEYS,
  ...LABEL_KEYS,
  "prediction",
]);

const firstValue = (row: MetadataRow, keys: string[]): string => {
  for (const key of keys) {
    const value = row[key];
    if (value !== undefined && value !== null && String(value).trim()) {
      return String(value).trim();
    }
  }
  return "";
};

const recordId = (row: MetadataRow, index: number): string =>
  firstValue(row, ID_KEYS) || String(index);

const displayPercent = (value: number): string => `${(value * 100).toFixed(1)}%`;

export const FailureDiscoveryPanel = ({
  model,
  dataset,
  predictionMap,
}: FailureDiscoveryPanelProps) => {
  const [metadata, setMetadata] = useState<MetadataRow[]>([]);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DiscoveryResult | null>(null);

  useEffect(() => {
    setResult(null);
    setError(null);
    if (!dataset || dataset === "custom") {
      setMetadata([]);
      setMetadataError(null);
      return;
    }

    const controller = new AbortController();
    setIsLoadingMetadata(true);
    setMetadataError(null);
    fetch(`${API_BASE}/${encodeURIComponent(dataset)}/metadata`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then(async response => {
        if (!response.ok) throw new Error(`Metadata request failed: ${response.status}`);
        return response.json();
      })
      .then(data => setMetadata(Array.isArray(data) ? data : []))
      .catch(caught => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setMetadataError(caught instanceof Error ? caught.message : "Unable to load metadata");
      })
      .finally(() => setIsLoadingMetadata(false));

    return () => controller.abort();
  }, [dataset]);

  const task = model?.startsWith("whisper") ? "transcription" : "classification";
  const records = useMemo(() => metadata.flatMap((row, index) => {
    const id = recordId(row, index);
    const prediction = predictionMap[id];
    const groundTruth = firstValue(row, task === "transcription" ? TRANSCRIPT_KEYS : LABEL_KEYS);
    if (!prediction || !groundTruth) return [];

    const features = Object.fromEntries(
      Object.entries(row).filter(([key, value]) =>
        !EXCLUDED_FEATURE_KEYS.has(key.toLowerCase()) &&
        value !== null &&
        value !== undefined &&
        String(value).trim() !== ""
      )
    );
    return [{ id, prediction: String(prediction), ground_truth: groundTruth, features }];
  }), [metadata, predictionMap, task]);

  const runDiscovery = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/analysis/failure-discovery`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          task,
          records,
          min_slice_size: Math.max(5, Math.floor(records.length * 0.05)),
          max_depth: 3,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Failure discovery failed");
      setResult(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failure discovery failed");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <BrainCircuit className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">Automated Failure Discovery</h3>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Learns repeatable error patterns from completed predictions and dataset metadata.
          </p>
        </div>
        <Button
          size="sm"
          className="h-8 text-xs"
          onClick={runDiscovery}
          disabled={isRunning || isLoadingMetadata || records.length < 20}
        >
          <RefreshCw className={`mr-1.5 h-3 w-3 ${isRunning ? "animate-spin" : ""}`} />
          {isRunning ? "Analyzing" : "Discover"}
        </Button>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        <Badge variant="outline">{records.length} labeled predictions</Badge>
        <Badge variant="outline">{metadata.length} dataset records</Badge>
        <Badge variant="outline">{task}</Badge>
      </div>

      {records.length < 20 && !metadataError && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-xs">
            Run batch inference first. At least 20 records with predictions, ground truth, and metadata are required.
          </AlertDescription>
        </Alert>
      )}

      {(metadataError || error) && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-xs">{metadataError || error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <>
          <Card>
            <CardHeader className="p-3 pb-1">
              <CardTitle className="text-xs">Analysis quality</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-3 p-3 pt-1 text-xs">
              <div><span className="text-muted-foreground">Baseline error</span><div className="font-semibold">{displayPercent(result.baseline_error_rate)}</div></div>
              <div><span className="text-muted-foreground">Validation</span><div className="font-semibold">{result.validation.value.toFixed(3)}</div></div>
              <div><span className="text-muted-foreground">Findings</span><div className="font-semibold">{result.findings.length}</div></div>
            </CardContent>
          </Card>

          {result.message && <p className="text-xs text-muted-foreground">{result.message}</p>}

          {result.findings.map(finding => (
            <Card key={finding.id}>
              <CardHeader className="p-3 pb-1">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-xs leading-5">{finding.rules.join(" and ")}</CardTitle>
                  <Badge>{finding.error_lift?.toFixed(2)}×</Badge>
                </div>
              </CardHeader>
              <CardContent className="p-3 pt-1 text-xs">
                <div className="mb-1 flex justify-between text-muted-foreground">
                  <span>{finding.sample_count} examples</span>
                  <span>{displayPercent(finding.error_rate)} error</span>
                </div>
                <Progress value={Math.min(100, finding.error_rate * 100)} className="h-1.5" />
                <p className="mt-2 truncate text-[11px] text-muted-foreground" title={finding.example_ids.join(", ")}>
                  Examples: {finding.example_ids.join(", ")}
                </p>
              </CardContent>
            </Card>
          ))}

          {result.feature_importance.length > 0 && (
            <Card>
              <CardHeader className="p-3 pb-1"><CardTitle className="text-xs">Associated metadata</CardTitle></CardHeader>
              <CardContent className="space-y-2 p-3 pt-1">
                {result.feature_importance.slice(0, 6).map(item => (
                  <div key={item.feature}>
                    <div className="mb-1 flex justify-between text-[11px]"><span>{item.feature}</span><span>{displayPercent(item.importance)}</span></div>
                    <Progress value={item.importance * 100} className="h-1" />
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
};
