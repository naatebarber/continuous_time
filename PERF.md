# Performance betwen tiny networks

with parms:

```rust
    let mut a = tiny::SsmNetwork::new(24, 1, 1);
    a.weave(1.);
```

steps per second: ~1800

with parms:
```rust
    let mut a = tiny::HashNetwork::new(24, 1, 1);
    a.weave(1.);
```

steps per second: ~200

### result

Sparse strength matrix is far more powerful than hashnetwork. 
hash spends too much time fucking with hashmap inserts and stuff