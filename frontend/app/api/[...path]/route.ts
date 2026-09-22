/**
 * Server-side proxy: /api/<anything>  ->  <BACKEND_URL>/<anything>
 *
 * THIS IS A ROUTE HANDLER, NOT A `rewrites()` ENTRY, AND THE DIFFERENCE BIT.
 * `next build` evaluates `rewrites()` once and freezes the result into
 * `.next/routes-manifest.json`, so a BACKEND_URL set at `next start` is
 * silently ignored — the build's value wins. MEASURED: a build made with the
 * default and started with BACKEND_URL=…:8300 still proxied to :8000 and
 * returned FastAPI's 404 for every call. A route handler reads the environment
 * PER REQUEST, so the backend URL is a deploy-time setting rather than a
 * build-time one.
 *
 * Two other reasons the proxy exists at all:
 *   • NO CORS. `api/main.py` ships without CORS middleware and should not have
 *     to know a frontend origin exists; same-origin calls sidestep it.
 *   • ONE SETTING. Set BACKEND_URL, or nothing, and get the documented
 *     default.
 *
 * Paths are NOT renamed: /api/ask -> /ask. The prefix only separates proxied
 * calls from Next's own routes.
 */

import { NextRequest } from "next/server";

const BACKEND_URL = () => process.env.BACKEND_URL || "http://127.0.0.1:8000";

/** Never cache: corpus stats and ingest progress change under us. */
export const dynamic = "force-dynamic";

function targetUrl(request: NextRequest, path: string[]): string {
  const search = request.nextUrl.search;
  return `${BACKEND_URL()}/${path.join("/")}${search}`;
}

async function proxy(request: NextRequest, path: string[]): Promise<Response> {
  const url = targetUrl(request, path);

  // Hop-by-hop and origin-specific headers must not be forwarded: `host` would
  // point at the frontend, and a stale `content-length` breaks a streamed body.
  const headers = new Headers();
  for (const [key, value] of request.headers) {
    const k = key.toLowerCase();
    // `expect` MUST BE STRIPPED, AND THIS IS WHAT BROKE "ADD FILING".
    // A client sending a large body commonly adds `Expect: 100-continue` and
    // waits for the server to agree before sending it. undici refuses to
    // forward that header and throws `NotSupportedError: expect header not
    // supported` - surfaced as a bare "fetch failed" until the cause was
    // logged. It is a hop-by-hop negotiation between one client and one
    // server, so a proxy must terminate it rather than pass it on.
    //
    // MEASURED: a 10 MB filing failed in the same second it was sent while
    // a 30-byte POST through the identical path succeeded - because the small
    // body never triggered the header.
    if (
      k === "host" ||
      k === "connection" ||
      k === "content-length" ||
      k === "transfer-encoding" ||
      k === "expect"
    ) {
      continue;
    }
    headers.set(key, value);
  }

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  // The body is buffered rather than streamed with `duplex: "half"`. Streaming
  // was NOT what broke uploads - the `expect` header above was - so this is a
  // robustness preference, not a fix: a filing is capped at ~16 MB, so holding
  // one in memory per upload is cheap, and it lets `fetch` set content-length
  // itself instead of relying on chunked transfer being negotiated correctly.
  const body = hasBody ? await request.arrayBuffer() : undefined;

  try {
    const response = await fetch(url, {
      method: request.method,
      headers,
      body,
      redirect: "manual",
      // A question takes 60-130 s (5-9 sequential model calls, and possibly a
      // second retrieval tier). No timeout: aborting would be indistinguishable
      // from the system declining, and telling those apart is the whole product.
      signal: AbortSignal.timeout(15 * 60 * 1000),
    } as RequestInit);

    const out = new Headers(response.headers);
    out.delete("content-encoding");
    out.delete("content-length");
    return new Response(response.body, { status: response.status, headers: out });
  } catch (error) {
    // The backend being down is an operational fact the UI should state plainly,
    // not a blank screen.
    //
    // `error.cause` is reported too: undici's own message is the useless string
    // "fetch failed" for every network-layer failure, and chasing the upload bug
    // above cost time precisely because the real reason was hidden one level
    // down.
    const message = error instanceof Error ? error.message : String(error);
    const cause =
      error instanceof Error && error.cause ? ` <- ${String(error.cause)}` : "";
    return Response.json(
      {
        detail:
          `Cannot reach the backend at ${BACKEND_URL()}. Start it with ` +
          `"uvicorn analyst_copilot.api.main:app --port 8000 --app-dir src", ` +
          `or set BACKEND_URL. (${message}${cause})`,
      },
      { status: 502 },
    );
  }
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
export async function POST(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
export async function PUT(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
export async function DELETE(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
