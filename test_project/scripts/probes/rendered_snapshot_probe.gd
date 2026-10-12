extends Node2D
# Optional Windows/Godot 4.4.1 responsiveness proxy, not actual ReNemiSys gameplay.
var model: Node
var elapsed := 0.0
var since_decision := 0.0
var last_frame_us := 0
var baseline: Array[float] = []
var during: Array[float] = []
var batch_times: Array[float] = []
var phase := 0.0

func p95(values: Array[float]) -> float:
    values.sort()
    return values[clampi(ceili(values.size() * 0.95) - 1, 0, values.size() - 1)]

func _ready() -> void:
    if DisplayServer.get_name() == "headless":
        push_error("Rendered probe needs a real display")
        get_tree().quit(1)
        return
    var data = JSON.parse_string(FileAccess.get_file_as_string("res://scripts/probes/fixture_small_policy.json"))
    model = ClassDB.instantiate("NeuralNetworkNode")
    add_child(model)
    if not data is Dictionary or not model.load_snapshot(data):
        push_error("Bundled neural snapshot unavailable")
        get_tree().quit(1)
        return
    Engine.max_fps = 60

func _process(delta: float) -> void:
    var now_us := Time.get_ticks_usec()
    if last_frame_us != 0:
        var interval := float(now_us - last_frame_us) / 1000.0
        if elapsed < 2.0:
            baseline.append(interval)
        else:
            during.append(interval)
    last_frame_us = now_us
    elapsed += delta
    phase += delta
    since_decision += delta
    queue_redraw()
    if elapsed >= 2.0 and since_decision >= 0.1:
        since_decision = 0.0
        var observations := []
        for i in range(20):
            observations.append([float(i % 4) / 3.0, sin(phase + i)])
        var start_us := Time.get_ticks_usec()
        var output: Array = model.predict(observations)
        batch_times.append(float(Time.get_ticks_usec() - start_us) / 1000.0)
        if output.size() != 20:
            push_error("Prediction shape invalid during render")
            get_tree().quit(1)
            return
    if elapsed >= 14.0 and OS.get_environment("MLKIT_AUTORUN") == "1":
        print("MLKIT_RENDER display=", DisplayServer.get_name(), " rectangles=400",
            " baseline_frame_p95_ms=", p95(baseline), " inference_frame_p95_ms=", p95(during),
            " inference_frame_max_ms=", during.max(), " 20_agent_batch_p95_ms=", p95(batch_times))
        get_tree().quit()

func _draw() -> void:
    for i in range(400):
        draw_rect(Rect2(Vector2(float(i % 25) * 41.0 + sin(phase) * 5.0, float(i / 25) * 31.0), Vector2(34, 25)),
            Color(0.2, 0.3 + float(i % 5) * 0.1, 0.5))
