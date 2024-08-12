# CTRNN experimentation

Look in `tiny/`;

**ctrnn network types, ranked inversely by performance**

 - **HashNetwork**: each neuron holds connections to other neurons via raw pointers. theoretically was supposed to be very fast O(1) access to other neurons during forward and backward pass, but hashmaps are slow in this context and outweighed the benefits of direct access. lowest performance.
 - **PowerNetwork**: uses a centralized synaptic strength matrix, and neurons hold their own caches. neuron caches are serialized and sent to worker threads, splitting BPTT steps over multiple cores (which works since they arent dependent on one another). slow due to high serialization overhead, and because in this multithreaded implementation we pass one message per backprop time step.
 - **AsyncPowerNetwork**: same as PowerNetwork, except the step loop does not wait for the workers to finish their current backprop tasks, and instead applies gradients as they come in. slightly faster due to the lack of polling, but still pretty slow, and the async gradient application could cause unpredictable behavior in the network.
 - **SSMNetwork**: A single-threaded network, holding a synaptic strength matrix centrally. very fast
 - **BulkPowerNetwork**: A derivative of the PowerNetwork, but instead of sending a zmq message for every BPTT step, they are batched. this way each worker only needs to receive and send one message per train step. about 3x faster than SSMNetwork on normal loads, and far faster when the complexity of the network grows.

all *PowerNetwork derivates make use of ZMQ, so they are inherently distributable. so far **BulkPowerNetwork** shows the most promise.

## Proofs of concept:

early python experiments to create basic architectures

### run v1:

shell 1, spawn the feed generator:
```bash
cd waitress
cargo run -- -f waitress.config.json
```

shell 2, attach ctrnn env:
```bash
cd proto
python env.py
```

### run v2:

shell 1, spawn the feed generator:
```bash
cd waitress
cargo run -- -f waitress.config.json
```

shell 2, attach ctrnn2 env:
```bash
cd proto
python main.py
```


