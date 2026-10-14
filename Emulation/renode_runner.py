"""Renode Headless Runner & Multi-Node Controller for PBL SCADA System.

Manages the lifecycle of Renode for virtual controller execution and bridges
virtual serial/socket terminals for deterministic automated validation.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("renode_runner")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESC_PATH = BASE_DIR / "Emulation" / "renode" / "plc_node.resc"
DEFAULT_RENODE_PATH = Path(r"C:\Tools\renode\renode.exe")


class RenodeController:
    """Controls the Renode emulation lifecycle for headless automated testing."""

    def __init__(
        self,
        resc_path: Optional[str | Path] = None,
        renode_executable: Optional[str | Path] = None,
        port: int = 15503,
        startup_timeout: float = 10.0,
    ):
        self.resc_path = Path(resc_path).resolve() if resc_path else DEFAULT_RESC_PATH
        self.renode_path = (
            Path(renode_executable).resolve()
            if renode_executable
            else self._find_renode()
        )
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None

    def _find_renode(self) -> Path:
        candidates = [
            DEFAULT_RENODE_PATH,
            Path(r"C:\Program Files\Renode\bin\renode.exe"),
            Path(shutil.which("renode.exe") or ""),
        ]
        for c in candidates:
            if c and c.is_file():
                return c
        return DEFAULT_RENODE_PATH

    @classmethod
    def kill_all_zombies(cls):
        """Clean up any dangling Renode processes."""
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "renode.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "Renode.exe"], capture_output=True)

    def start(self) -> bool:
        self.kill_all_zombies()

        if not self.renode_path.exists():
            logger.warning(
                "Renode binary not found at %s. Use system_emulator.py for full multi-node emulation.",
                self.renode_path,
            )
            return False

        if not self.resc_path.exists():
            logger.warning("Renode .resc script not found at %s", self.resc_path)
            return False

        cmd = [
            str(self.renode_path),
            "--plain",
            "--disable-gui",
            "-P",
            "-1",
            "-e",
            f"$port={self.port}; i @{self.resc_path.as_posix()}",
        ]

        logger.info("Spawning Renode process: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )

        atexit.register(self.stop)
        return self._wait_for_port()

    def _wait_for_port(self) -> bool:
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", self.port))
                s.close()
                return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(0.2)
        return False

    def stop(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        capture_output=True,
                    )
                else:
                    self.process.kill()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing RenodeController...")
    controller = RenodeController()
    if controller.start():
        print(
            f"[OK] Renode running and socket listening on 127.0.0.1:{controller.port}"
        )
        time.sleep(1.0)
        controller.stop()
        print("[OK] Renode stopped cleanly.")
    else:
        print("[INFO] Renode standalone controller ready.")
