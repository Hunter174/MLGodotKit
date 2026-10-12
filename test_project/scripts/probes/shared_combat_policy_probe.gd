extends SceneTree
# Issue #50: isolated API/inference probe. Random weights, no combat quality claim.
# Run: Godot 4.4.1 --headless --path test_project --script res://scripts/probes/shared_combat_policy_probe.gd
func percentile(samples: Array[float], fraction: float) -> float:
    samples.sort()
    return samples[clampi(ceili(float(samples.size()) * fraction) - 1, 0, samples.size() - 1)]

func _initialize() -> void:
    call_deferred("run")

func run() -> void:
    if not ClassDB.class_exists("NeuralNetworkNode"):
        push_error("MLGodotKit extension missing")
        quit(1)
        return
    var model: Node = ClassDB.instantiate("NeuralNetworkNode")
    root.add_child(model)
    model.add_layer(16, 32, "relu")
    model.add_layer(32, 8, "linear")
    var copy: Node = ClassDB.instantiate("NeuralNetworkNode")
    root.add_child(copy)
    copy.add_layer(16, 32, "relu")
    copy.add_layer(32, 8, "linear")
    copy.copy_weights(model)
    var observations := []
    for i in range(20):
        var row := []
        for j in range(16):
            row.append(float((i * 3 + j) % 15) / 7.0 - 1.0)
        observations.append(row)
    var one := [observations[0]]
    var baseline: Array = model.predict(one)
    if baseline.size() != 1 or baseline[0].size() != 8 or baseline != copy.predict(one):
        push_error("Output shape or weight-copy regression")
        quit(1)
        return
    var single: Array[float] = []
    var twenty: Array[float] = []
    var batch: Array[float] = []
    for i in range(100):
        model.predict(one)
        model.predict(observations)
    for i in range(1000):
        var start := Time.get_ticks_usec()
        var output: Array = model.predict(one)
        single.append(float(Time.get_ticks_usec() - start) / 1000.0)
        if output.size() != 1 or output[0].size() != 8:
            push_error("Single prediction has wrong shape")
            quit(1)
            return
    for i in range(300):
        var start := Time.get_ticks_usec()
        for row in observations:
            model.predict([row])
        twenty.append(float(Time.get_ticks_usec() - start) / 1000.0)
        start = Time.get_ticks_usec()
        var output: Array = model.predict(observations)
        batch.append(float(Time.get_ticks_usec() - start) / 1000.0)
        if output.size() != 20:
            push_error("Batched prediction has wrong shape")
            quit(1)
            return
    if model.predict(one) != baseline:
        push_error("Prediction mutated the model")
        quit(1)
        return
    print("SHARED_POLICY_PROBE 16-32-8 single_p50_ms=", percentile(single, 0.5),
        " single_p95_ms=", percentile(single, 0.95), " single_p99_ms=", percentile(single, 0.99),
        " 20_individual_p95_ms=", percentile(twenty, 0.95),
        " 20_batch_p95_ms=", percentile(batch, 0.95), " copied_weights_equal=true")
    quit()
