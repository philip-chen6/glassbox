import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const execFileAsync = promisify(execFile);

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RunTraceBody = {
  prompt?: string;
  model?: string;
  useToy?: boolean;
  includeHidden?: boolean;
  includeAttention?: boolean;
  maxNewTokens?: number;
};

function normalizeBody(body: RunTraceBody): Required<RunTraceBody> {
  return {
    prompt: String(body.prompt ?? "").trim(),
    model: String(body.model ?? "distilgpt2"),
    useToy: Boolean(body.useToy ?? false),
    includeHidden: Boolean(body.includeHidden ?? true),
    includeAttention: Boolean(body.includeAttention ?? true),
    maxNewTokens: Math.max(0, Math.floor(Number(body.maxNewTokens ?? 24)))
  };
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const body = normalizeBody((await request.json()) as RunTraceBody);
    if (!body.prompt) {
      return NextResponse.json({ error: "Prompt is required." }, { status: 400 });
    }

    const repoRoot = path.resolve(process.cwd(), "..");
    const args = [
      "-m",
      "glassbox.cli",
      "--prompt",
      body.prompt,
      "--model",
      body.model,
      "--max-new-tokens",
      String(body.maxNewTokens)
    ];
    if (body.useToy) {
      args.push("--use-toy");
    }
    if (body.includeHidden) {
      args.push("--include-hidden");
    }
    if (body.includeAttention) {
      args.push("--include-attention");
    }

    const { stdout, stderr } = await execFileAsync("python3", args, {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONPATH: path.join(repoRoot, "src")
      },
      maxBuffer: 1024 * 1024 * 200
    });

    if (stderr && stderr.trim().length > 0) {
      // Keep stderr available for debugging without failing successful runs.
      console.warn(stderr);
    }

    const trace = JSON.parse(stdout);
    return NextResponse.json({ trace });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to generate trace.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
