# dicking around with CTRNNs

maybe something useful will come outta this but im mainly here to learn

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

### run v3
```bash

```