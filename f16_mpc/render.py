import socket
import json
import time
import multiprocessing

import numpy as np

from pyf16 import CoreOutput, Control, StateExtend, State


class F16TcpRender:
    def __init__(self, render_fps: int, host: str, port: int) -> None:
        self.render_fps = render_fps
        self.host = host
        self.port = port
        self.process = multiprocessing.Process(target=self._run)
        self.queue = multiprocessing.Queue()
        self.process.start()

    def _run(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Connecting to {self.host}:{self.port}")
            client_socket.connect((self.host, self.port))
        except socket.error as e:
            print(f"Failed to connect to {self.host}:{self.port} - {e}")
            self.queue.put(False)
            return
        self.queue.put(True)
        while True:
            data = self.queue.get()
            if data is None:
                break
            message = json.dumps(data)
            message += "\n"
            client_socket.sendall(message.encode("utf-8"))
        client_socket.close()

    def get(self):
        return self.queue.get()

    def render(
        self,
        output,
    ):
        self.queue.put(output)
        time.sleep(1 / self.render_fps)

    def close(self):
        self.queue.put(None)
        self.process.join()
