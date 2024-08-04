import zmq
from threading import Thread
from collections import deque
import json
from typing import TypedDict, Dict, List
import time

from graph_thread import GraphThread


class Frame(TypedDict):
    ts: float
    channels: Dict[str, float]
    
    
class OutputFrame(TypedDict):
    tau: float
    outputs: List[float]
    loss: float
    
    
class StepFrame(TypedDict):
    tau: float
    inputs: List[float]
    targets: List[float]
    
    
class InfFrame(TypedDict):
    tau: float
    inputs: List[float]
    

class TransferThread(Thread):
    def __init__(self, inputs: deque, outputs: deque, losses: deque):
        super().__init__()
        
        self.inputs = inputs
        self.outputs = outputs
        self.losses = losses
        
        self.context = zmq.Context()
        self.sub_sock = self.context.socket(zmq.SUB)
        self.push_sock = self.context.socket(zmq.PUSH)
        self.pull_sock = self.context.socket(zmq.PULL)
        
        self.sub_sock.subscribe(b"waitress")
        
        self.sub_sock.connect("tcp://127.0.0.1:1201")
        self.push_sock.connect("tcp://127.0.0.1:1101")
        self.pull_sock.connect("tcp://127.0.0.1:1102")
        
        self.poller = zmq.Poller()
        
        self.poller.register(self.sub_sock, zmq.POLLIN)
        self.poller.register(self.pull_sock, zmq.POLLIN)
        
    def run(self):
        recent_frames: deque[Frame] = deque()
        recent_outputs: deque[OutputFrame] = deque()
        
        input_ct = 0
        output_ct = 0
        
        while True:
            try:
                socks = dict(self.poller.poll(10))
                                
                if self.sub_sock in socks:
                    # drain the socket
                    while True:
                        try:
                            _, _, msgb = self.sub_sock.recv_multipart(
                                flags=zmq.DONTWAIT
                            )
                            frame: Frame = { ** json.loads(msgb.decode("utf-8")) }
                            self.inputs.append((
                                input_ct, 
                                frame.get("channels").get("waitress") 
                            ))
                            input_ct += 1
                            recent_frames.append(frame)
                        except Exception as e:
                            break
                
                if self.pull_sock in socks:
                    cmd, msgb = self.pull_sock.recv_multipart()
                    if cmd.startswith(b"out"):
                        recent_outputs.append({ **json.loads(msgb.decode("utf-8")) })
                        
                while len(recent_frames) > 1:
                    input_frame = recent_frames.popleft()
                    target_frame = recent_frames[0]
                    
                    step_frame: StepFrame = {
                        "inputs": [ *input_frame.get("channels").values() ],
                        "targets": [ *target_frame.get("channels").values() ],
                        "tau": time.time()
                    }
                    
                    msgb = json.dumps(step_frame).encode("utf-8")
                    
                    self.push_sock.send_multipart([b"step", msgb])
                    
                for output_frame in recent_outputs:
                    self.losses.append((output_ct, output_frame.get("loss")))
                    self.outputs.append((output_ct, output_frame.get("outputs")[0]))
                    print(output_frame)
                    output_ct += 1
                    
            except Exception as e:
                print(e)
        
def main():
    inputs = deque()
    outputs = deque()
    losses = deque()
    
    tt = TransferThread(inputs=inputs, outputs=outputs, losses=losses)
    gt = GraphThread([ inputs, outputs ])
    
    tt.daemon = True
    tt.start()
    
    gt.run_blocking()
    
main()