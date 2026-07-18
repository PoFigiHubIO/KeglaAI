/**
 * scripts/cf_worker.js
 * 
 * Cloudflare Worker for static routing with Cloudflare KV.
 * 
 * Exposes:
 *   - GET /active      : Returns the current active LLM and Media backend URLs.
 *   - POST /register   : Updates the active backend URLs in Cloudflare KV.
 *                        Requires Authorization header: Bearer <secret>
 *   - GET/POST/etc /*  : Proxies all other requests to the active LLM URL.
 * 
 * KV Binding:
 *   Bind a Cloudflare KV namespace named "KAGGLING" to the variable "KAGGLING".
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // --- Handle CORS preflight ---
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
      });
    }

    // --- GET /active ---
    if (url.pathname === "/active" && request.method === "GET") {
      const llm = await env.KAGGLING.get("ACTIVE_LLM_URL");
      const media = await env.KAGGLING.get("ACTIVE_MEDIA_URL");
      return new Response(JSON.stringify({ llm, media }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        }
      });
    }

    // --- POST /register ---
    if (url.pathname === "/register" && request.method === "POST") {
      try {
        const authHeader = request.headers.get("Authorization");
        const expectedSecret = env.HANDOVER_SECRET || "default_secret";
        
        if (!authHeader || authHeader !== `Bearer ${expectedSecret}`) {
          return new Response(JSON.stringify({ error: "Unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" }
          });
        }

        const body = await request.json();
        const llmUrl = body.llm_url;
        const mediaUrl = body.media_url;

        if (llmUrl) {
          await env.KAGGLING.put("ACTIVE_LLM_URL", llmUrl);
          console.log(`Registered ACTIVE_LLM_URL: ${llmUrl}`);
        }
        if (mediaUrl) {
          await env.KAGGLING.put("ACTIVE_MEDIA_URL", mediaUrl);
          console.log(`Registered ACTIVE_MEDIA_URL: ${mediaUrl}`);
        }

        return new Response(JSON.stringify({ success: true }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 500,
          headers: { "Content-Type": "application/json" }
        });
      }
    }

    // --- Proxy requests to the Active LLM Backend URL ---
    try {
      const activeLlm = await env.KAGGLING.get("ACTIVE_LLM_URL");
      if (!activeLlm) {
        return new Response(JSON.stringify({ error: "No active LLM backend registered in KV." }), {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        });
      }

      // Construct target URL
      const targetUrl = new URL(url.pathname + url.search, activeLlm);
      
      // Clone headers and modify Host header
      const headers = new Headers(request.headers);
      headers.set("Host", targetUrl.host);

      // Clone the request for proxying
      const proxyRequest = new Request(targetUrl, {
        method: request.method,
        headers: headers,
        body: request.body,
        redirect: "manual"
      });

      const response = await fetch(proxyRequest);
      
      // Add CORS headers to the response
      const newHeaders = new Headers(response.headers);
      newHeaders.set("Access-Control-Allow-Origin", "*");
      
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: `Proxy failed: ${err.message}` }), {
        status: 502,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        }
      });
    }
  }
};
