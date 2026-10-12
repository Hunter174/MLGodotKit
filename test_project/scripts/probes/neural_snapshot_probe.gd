extends SceneTree
# Local native neural artifact regression; test input is bundled JSON in res://.
func check(condition: bool, explanation: String) -> void:
    if not condition:
        push_error(explanation)
        failures += 1

var failures := 0

func _initialize() -> void:
    call_deferred("run")

func run() -> void:
    if not ClassDB.class_exists("NeuralNetworkNode"):
        check(false, "NeuralNetworkNode GDExtension unavailable")
        quit(1)
        return
    var model: Node = ClassDB.instantiate("NeuralNetworkNode")
    root.add_child(model)
    model.add_layer(2, 16, "relu")
    model.add_layer(16, 2, "linear")
    var input := [[0.25, 0.75]]
    var original: Array = model.predict(input)
    var config: Dictionary = model.get_snapshot()
    var unsupported: Node = ClassDB.instantiate("NeuralNetworkNode")
    root.add_child(unsupported)
    unsupported.add_layer(2, 2, "unknown_activation")
    check(unsupported.get_snapshot().is_empty(), "Unsupported activation must not export an unusable snapshot")
    check(config.version == 1 and config.layers.size() == 2 and model.get_layers().size() == 2,
        "Architecture snapshot must match add_layer model")
    var restored: Node = ClassDB.instantiate("NeuralNetworkNode")
    root.add_child(restored)
    check(restored.load_snapshot(config), "In-memory snapshot load failed")
    check(restored.predict(input) == original, "In-memory snapshot changed inference")
    var artifact := "user://neural_snapshot_probe.json"
    var snapshot_text := JSON.stringify(config)
    var envelope := {"schema": 1, "snapshot_json": snapshot_text, "sha256": snapshot_text.sha256_text()}
    var save := FileAccess.open(artifact, FileAccess.WRITE)
    check(save != null, "Snapshot file open failed")
    if save != null:
        save.store_string(JSON.stringify(envelope))
        save.close()
    var persisted = JSON.parse_string(FileAccess.get_file_as_string(artifact))
    check(persisted is Dictionary and persisted.snapshot_json.sha256_text() == persisted.sha256,
        "External artifact checksum mismatch")
    var serialized = JSON.parse_string(persisted.snapshot_json)
    check(serialized is Dictionary and restored.load_snapshot(serialized), "JSON artifact reload failed")
    var corrupted_envelope: Dictionary = persisted.duplicate(true)
    corrupted_envelope.snapshot_json += " "
    check(corrupted_envelope.snapshot_json.sha256_text() != corrupted_envelope.sha256,
        "Tampered artifact must be rejected before model loading")
    var loaded_output: Array = restored.predict(input)
    for i in range(2):
        check(absf(float(loaded_output[0][i]) - float(original[0][i])) < 0.00001,
            "JSON roundtrip exceeded prediction tolerance")
    var corrupted: Dictionary = config.duplicate(true)
    var entries: Array = corrupted.layers
    var first: Dictionary = entries[0]
    var rows: Array = first.weights
    var row: Array = rows[0]
    row[0] = NAN
    check(not restored.load_snapshot(corrupted) and restored.predict(input) == loaded_output,
        "Nonfinite weights must fail without changing loaded model")
    corrupted = config.duplicate(true)
    corrupted.layers[1].output_size = 9999
    check(not restored.load_snapshot(corrupted) and restored.predict(input) == loaded_output,
        "Incompatible geometry must fail without changing loaded model")
    corrupted = config.duplicate(true)
    corrupted.optimizer = "unrecognized"
    check(not restored.load_snapshot(corrupted) and restored.predict(input) == loaded_output,
        "Invalid optimizer must fail without changing loaded model")
    var fixture_path := "res://scripts/probes/fixture_small_policy.json"
    check(FileAccess.file_exists(fixture_path), "Bundled res:// fixture missing")
    var fixture = JSON.parse_string(FileAccess.get_file_as_string(fixture_path))
    check(fixture is Dictionary and restored.load_snapshot(fixture), "Bundled fixture load failed")
    var expected: Array = restored.predict([[1.0, 2.0]])
    check(expected.size() == 1 and expected[0].size() == 2 and absf(float(expected[0][0]) - 7.5) < 0.00001
        and absf(float(expected[0][1]) - 9.0) < 0.00001, "Bundled fixture predicts wrong values")
    print("NEURAL_SNAPSHOT_PROBE ", "PASSED" if failures == 0 else "FAILED", " checks=roundtrip,geometry,nonfinite,atomicity,res_asset")
    quit(1 if failures else 0)
