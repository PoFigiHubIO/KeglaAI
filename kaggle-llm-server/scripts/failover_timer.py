#!/usr/bin/env python3
"""
scripts/failover_timer.py

Uptime monitor and rotation trigger.
Runs in the background, waits for 8 hours 50 minutes (by default),
then configures target credentials and runs 'kaggle kernels push'
to boot the next instance in the failover ring.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [failover-timer]: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("failover_timer")

# Configuration (defaults to 8 hours 50 minutes)
try:
    from kaggle_secrets import UserSecretsClient
    ROTATION_TIME_SECONDS = int(UserSecretsClient().get_secret("ROTATION_TIME_SECONDS"))
    log.info(f"Loaded ROTATION_TIME_SECONDS from Kaggle Secrets: {ROTATION_TIME_SECONDS}s")
except Exception:
    ROTATION_TIME_SECONDS = int(os.environ.get("ROTATION_TIME_SECONDS", str(8 * 3600 + 50 * 60)))


async def trigger_next_node():
    log.info("Starting next node trigger protocol...")

    username = os.environ.get("NEXT_KAGGLE_USERNAME", "")
    key = os.environ.get("NEXT_KAGGLE_KEY", "")
    slug = os.environ.get("NEXT_KAGGLE_SLUG", "keglaai")

    if not username or not key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            if not username:
                username = user_secrets.get_secret("NEXT_KAGGLE_USERNAME")
            if not key:
                key = user_secrets.get_secret("NEXT_KAGGLE_KEY")
            log.info("Loaded next Kaggle credentials from secrets.")
        except Exception as e:
            log.warning(f"Could not load credentials from Kaggle Secrets: {e}")

    if not username or not key:
        log.warning("NEXT_KAGGLE_USERNAME or NEXT_KAGGLE_KEY not set. Cannot push next kernel. Handover aborted.")
        return False

    # Step 1: Write ~/.kaggle/kaggle.json
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    with open(kaggle_json_path, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    
    # Enforce safe permissions on Unix
    if os.name != "nt":
        os.chmod(kaggle_json_path, 0o600)
    log.info(f"Wrote Kaggle credentials for next user '{username}'.")

    # Step 2: Create temp directory for metadata
    meta_dir = "./data/standby_metadata"
    os.makedirs(meta_dir, exist_ok=True)
    
    # Kaggle API metadata
    meta = {
        "id": f"{username}/{slug}",
        "title": "Kaggle LLM Server",
        "code_file": "/kaggle/working/KeglaAI/kaggle-llm-server/notebooks/kaggle_run_all.ipynb",
        "language": "notebook",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": []
    }
    
    with open(os.path.join(meta_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Wrote kernel metadata for kernel '{username}/{slug}'.")

    # Step 3: Run kaggle kernels push
    log.info(f"Executing: kaggle kernels push -p {meta_dir}")
    try:
        # Run subprocess
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", meta_dir],
            capture_output=True, text=True, timeout=120
        )
        log.info(f"Kaggle API stdout: {result.stdout}")
        if result.returncode == 0:
            log.info("✅ Next node successfully triggered!")
            return True
        else:
            log.error(f"❌ Kaggle API command failed (code={result.returncode}): {result.stderr}")
            return False
    except Exception as e:
        log.error(f"❌ Error executing kaggle kernels push: {e}", exc_info=True)
        return False


async def main():
    log.info(f"Failover timer active. Sleep duration: {ROTATION_TIME_SECONDS} seconds (~{ROTATION_TIME_SECONDS / 3600:.1f} hours).")
    
    # Wait for the rotation time limit
    await asyncio.sleep(ROTATION_TIME_SECONDS)
    
    log.info("⏰ Sleep limit reached! Initiating failover handover...")
    
    # Attempt to trigger the next node in the ring
    success = await trigger_next_node()
    if success:
        log.info("Next node is booting up. Current node will poll cloud storage and local signals for handover...")
        
        # Poll for handover signal (every 10 seconds, up to 15 minutes)
        for i in range(90):
            await asyncio.sleep(10)
            
            # Check local handover flag (written if HTTP endpoint succeeds)
            if os.path.exists("/tmp/handover_complete"):
                log.info("Local handover complete flag detected. Exiting timer.")
                break
                
            # Check cloud storage signal (fallback for Cloudflare Tunnels direct mode)
            try:
                res = subprocess.run(
                    ["bash", "scripts/rclone_sync.sh", "check_signal"],
                    capture_output=True, text=True, timeout=15
                )
                if res.returncode == 0:
                    log.info("🔔 Handover signal file detected on cloud storage!")
                    
                    # Write local handover_complete file to shut down keep-alive cell
                    with open("/tmp/handover_complete", "w") as f:
                        f.write("handover complete via cloud signal")
                    
                    # Delete the signal file from cloud so it's clean for next node
                    subprocess.run(
                        ["bash", "scripts/rclone_sync.sh", "delete_signal"],
                        capture_output=True, text=True, timeout=15
                    )
                    log.info("✅ Remote handover signal cleared. Exiting.")
                    break
            except Exception as e:
                log.warning(f"Error checking cloud signal: {e}")
    else:
        log.warning("Failover trigger failed. Uptime loop will continue to prevent crash/shutdown.")


if __name__ == "__main__":
    asyncio.run(main())
