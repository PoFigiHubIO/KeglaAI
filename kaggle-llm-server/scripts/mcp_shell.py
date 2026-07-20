#!/usr/bin/env python3
import sys
import json
import subprocess
import os

def log(msg):
    sys.stderr.write(f"[mcp-shell] {msg}\n")
    sys.stderr.flush()

def main():
    log("Starting stdio MCP Shell Server...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            data = json.loads(line)
            method = data.get("method")
            req_id = data.get("id")
            
            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "mcp-shell",
                            "version": "1.0.0"
                        }
                    }
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "execute_command",
                                "description": "Выполнить произвольную shell-команду в рабочей директории Kaggle и вернуть stdout/stderr",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string", "description": "Команда для исполнения (например, 'pip install ...' или 'python scripts/...')"}
                                    },
                                    "required": ["command"]
                                }
                            }
                        ]
                    }
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                
            elif method == "tools/call":
                params = data.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                if name == "execute_command":
                    command = arguments.get("command", "")
                    log(f"Executing command: {command}")
                    
                    try:
                        # Execute in shell
                        res = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=180
                        )
                        output = f"Exit code: {res.returncode}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
                    except subprocess.TimeoutExpired:
                        output = "Error: Command timed out after 180 seconds."
                    except Exception as e:
                        output = f"Error executing command: {e}"
                        
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": output
                                }
                            ]
                        }
                    }
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            else:
                # Standard empty response or ignore notifications
                if req_id is not None:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method {method} not found"
                        }
                    }
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
