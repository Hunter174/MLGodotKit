extends Node
# Run as the main scene in an isolated debug export, not under --script.
func _ready() -> void:
    var fixture := "res://scripts/probes/fixture_small_policy.json"
    if not FileAccess.file_exists(fixture) or not ClassDB.class_exists("NeuralNetworkNode"):
        push_error("Bundled snapshot or native extension missing")
        get_tree().quit(1)
        return
    var model: Node = ClassDB.instantiate("NeuralNetworkNode")
    add_child(model)
    var data = JSON.parse_string(FileAccess.get_file_as_string(fixture))
    if not data is Dictionary or not model.load_snapshot(data):
        push_error("Bundled snapshot did not load")
        get_tree().quit(1)
        return
    var result: Array = model.predict([[1.0, 2.0]])
    if result.size() != 1 or result[0].size() != 2 or absf(float(result[0][0]) - 7.5) > 0.00001 or absf(float(result[0][1]) - 9.0) > 0.00001:
        push_error("Bundled snapshot produced incorrect outputs")
        get_tree().quit(1)
        return
    print("NEURAL_EXPORT_PROBE_PASSED")
    get_tree().quit()
