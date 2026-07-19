# Detailed Configuration & 10-Minute Verification Guide

This guide details how to configure external integrations (Ngrok, Cloudflare Worker, Yandex Disk WebDAV) and set up a 10-minute loop rotation test between two Kaggle accounts to verify zero-downtime database synchronization and automatic shutdown.

---

## 1. Setting Up External Services

### A. Ngrok (Tunneling)
1. Log in to [ngrok.com](https://ngrok.com/).
2. In the left-hand sidebar menu, navigate to **Getting Started > Your Authtoken**.
3. Copy the token string (e.g. `2Rj...`).
4. (Optional) If you have a free static domain, go to **Cloud Edge > Domains** and copy your domain string (e.g., `sabbath-lucid-molar.ngrok-free.dev`).
5. *For multi-GPU concurrent runs, using two different Ngrok tokens (Account A and Account B) prevents port-binding conflicts during the short handover overlap period.*

### B. Cloudflare Worker (Traffic Router)
1. Log in to [dash.cloudflare.com](https://dash.cloudflare.com/).
2. In the sidebar, select **Workers & Pages**.
3. Create a KV Namespace:
   - Click **KV** on the sub-menu.
   - Click **Create a Namespace**. Name it `KAGGLER_ROUTER_KV`.
4. Create the Worker:
   - Go back to **Workers & Pages > Overview**.
   - Click **Create Application > Create Worker**.
   - Name it `kaggle-router` and click **Deploy**.
   - Click **Edit Code**, paste the entire contents of **[cf_worker.js](file:///d:/123VsakayaVsyachina/___LAB/AI_WEB/_RestrictAI/Kaggle/KeglaAI/kaggle-llm-server/scripts/cf_worker.js)**, and click **Save and Deploy**.
5. Bind KV and Secret to Worker:
   - Go to the `kaggle-router` Worker dashboard, select the **Settings** tab.
   - Under **KV Namespace Bindings**, click **Add binding**. Set Variable Name = `KAGGLER_ROUTER`, select KV Namespace = `KAGGLER_ROUTER_KV`. Click **Save**.
   - Under **Environment Variables**, click **Add variable**. Set Key = `HANDOVER_SECRET`, Value = `my_super_secret_token_123` (this must match the secret in your Kaggle Notebooks). Click **Save**.
6. Copy the Worker's public URL (e.g. `https://kaggle-router.my-username.workers.dev`).

### C. Yandex Disk (WebDAV DB Synchronization)
1. Log in to Yandex.
2. Go to App Passwords: [passport.yandex.ru/profile/phones](https://passport.yandex.ru/profile/phones) -> **Passwords and authorization** -> **App passwords**.
3. Click **Create app password**, select **Files (Disk)** as type, and name it `KaggleSync`.
4. Copy the generated 16-character password (e.g., `abcd-efgh-ijkl-mnop`).
5. Your login details: Provider = `yandex`, User = your Yandex email, Password = App Password.

---

## 2. Kaggle Secrets Configuration (2-Account Setup)

Let's label the two Kaggle accounts as **Account A** and **Account B**.
1. Create a notebook on both accounts named exactly `keglaai` (so the URL slug is `/keglaai`).
2. Go to **Settings > API** on both accounts and click **Create New Token**. This downloads `kaggle.json`. Open the file to extract the `username` and `key` for both accounts.
3. Open your Notebook Editor, go to **Add-ons > Secrets** and enter the following keys.

### Secrets Mapping Table

| Secret Key | Value on **Account A** | Value on **Account B** | Description |
| :--- | :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | *Same Telegram Bot Token* | *Same Telegram Bot Token* | Bot API Token from @BotFather |
| `HANDOVER_SECRET` | `my_super_secret_token_123` | `my_super_secret_token_123` | Secret matching CF Worker |
| `CF_WORKER_URL` | `https://kaggle-router.my-username.workers.dev` | `https://kaggle-router.my-username.workers.dev` | CF Worker URL |
| `RCLONE_PROVIDER` | `yandex` | `yandex` | Set to `yandex` |
| `RCLONE_USER` | `email@yandex.ru` | `email@yandex.ru` | Yandex email address |
| `RCLONE_PASS` | `abcd-efgh-ijkl-mnop` | `abcd-efgh-ijkl-mnop` | Yandex Disk App Password |
| `NGROK_AUTHTOKEN` | *Account A Ngrok Token 1* | *Account B Ngrok Token 1* | Tunnel auth for Port 8080 |
| `NGROK_AUTHTOKEN_2` | *Account A Ngrok Token 2* | *Account B Ngrok Token 2* | Tunnel auth for Port 8081 |
| `ROTATION_TIME_SECONDS` | `600` | `600` | **10-minute rotation override** |
| `NEXT_KAGGLE_USERNAME` | `account_b_username` | `account_a_username` | **Target account username** |
| `NEXT_KAGGLE_KEY` | `account_b_api_key` | `account_a_api_key` | **Target account API Key** |
| `NEXT_KAGGLE_SLUG` | `keglaai` | `keglaai` | Notebook slug (`keglaai`) |

---

## 3. 10-Minute Loop Verification Flow

Follow these steps to verify that the database syncs, traffic reroutes, and billing stops:

1. **Start Account A**: Manually start the notebook on Account A.
2. **First Boot**: Account A pulls the repository, downloads the DB from Yandex Disk (creates a fresh one if missing), starts LLM on GPU 0, starts Media Server + Bot + Timer on GPU 1, and registers the URLs with the Cloudflare Worker.
3. **Bot Handshake**: Open Telegram and send `/start` or `/mcp_list` to your bot. It will respond. Send `/status` to verify that both LLM and Media Server are reported active.
4. **Active Timer**: The background timer starts sleeping for 10 minutes (600s).
5. **Next Node Boot**: At 10 minutes, the timer on Account A wakes up, writes Account B's credentials into `~/.kaggle/kaggle.json`, and triggers `kaggle kernels push` targeting `account_b/keglaai`.
6. **Account B Startup**: Account B's notebook boots up in the cloud automatically.
7. **DB Download**: Account B downloads the latest `agent.db` from Yandex Disk, preserving your Telegram history and whitelists.
8. **Worker Update**: Account B starts its services, registers its URLs in the Cloudflare KV store, and routes new traffic to itself.
9. **Handover Trigger**: Account B sends a POST request to `/v1/handover/complete` on Account A's media server.
10. **Account A Graceful Exit**: Account A receives the handover request:
    - It uploads the final SQLite `agent.db` to Yandex Disk (`rclone_sync.sh upload`).
    - It kills its background processes and creates the `/tmp/handover_complete` flag.
    - The keep-alive cell detects the flag, exits the loop, and stops the notebook session (stopping billing).
11. **Continuous Loop**: In 10 minutes, Account B will trigger Account A, repeating the loop. The transition is completely transparent to the user because all requests are routed through the Cloudflare Worker.
