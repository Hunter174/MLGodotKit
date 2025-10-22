extends Node
class_name NumericStabilityTester

@export var steps: int = 30
@export var amplify_input: bool = true
@export var gamma: float = 0.95
@export var lr: float = 1e-3

var nn: NNNode
var scale: float = 1.0
var input: Array

func _ready() -> void:
	print("\n=== Numeric Stability Tester (NNNode) ===")
	_init_network()
	_init_input()
	_run_test()

# ---------------------------------------------------------
# Initialize a simple test network
# ---------------------------------------------------------
func _init_network() -> void:
	nn = NNNode.new()
	nn.add_layer(4, 16, "relu")
	nn.add_layer(16, 16, "relu")
	nn.add_layer(16, 2, "linear")
	nn.set_learning_rate(lr)
	print("Network initialized:")
	nn.model_summary()

# ---------------------------------------------------------
# Generate starting input
# ---------------------------------------------------------
func _init_input() -> void:
	input = []
	for i in range(4):
		input.append(randf_range(-1.0, 1.0))
	print("Initial input:", input)

# ---------------------------------------------------------
# Main numeric stress test
# ---------------------------------------------------------
func _run_test() -> void:
	for step in range(steps):
		if amplify_input:
			scale *= 1.5
		var scaled_input := []
		for v in input:
			scaled_input.append(v * scale)

		# forward pass
		var q_vals: Array = _flatten_1d(nn.forward(scaled_input))
		var q_min := _array_min(q_vals)
		var q_max := _array_max(q_vals)

		print("")
		print("--- Step " + str(step + 1) + " ---")
		print("Input scale: " + _fmt_sci(scale))
		print("Q range: [" + _fmt_sci(q_min) + ", " + _fmt_sci(q_max) + "]")

		if _has_invalid(q_vals):
			push_error("❌ NaN or INF detected at step " + str(step + 1))
			break

		var a := _argmax(q_vals)
		var r := randf_range(-1.0, 1.0)
		var next_q: Array = _flatten_1d(nn.forward(scaled_input))
		var max_next_q := _array_max(next_q)
		var td_target := r + gamma * max_next_q
		var td_error := td_target - float(q_vals[a])

		# lightweight manual gradient step (simulate RL update)
		var grad := [0.0, 0.0]
		grad[a] = td_error
		nn.backward([grad])

		print("r: " + _fmt_sci(r) +
			" | TD_target: " + _fmt_sci(td_target) +
			" | TD_error: " + _fmt_sci(td_error))

		if abs(td_error) > 1e6 or abs(q_max) > 1e6:
			push_warning("⚠️ Magnitude explosion likely.")
			break

# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------
func _flatten_1d(arr: Array) -> Array:
	if arr.size() == 0:
		return []
	if typeof(arr[0]) == TYPE_ARRAY:
		return arr[0]
	return arr

func _array_min(arr: Array) -> float:
	var mn := INF
	for v in arr:
		mn = min(mn, float(v))
	return mn

func _array_max(arr: Array) -> float:
	var mx := -INF
	for v in arr:
		mx = max(mx, float(v))
	return mx

func _argmax(arr: Array) -> int:
	var best_idx := 0
	var best_val := -INF
	for i in range(arr.size()):
		var v := float(arr[i])
		if v > best_val:
			best_val = v
			best_idx = i
	return best_idx

func _has_invalid(arr: Array) -> bool:
	for v in arr:
		if is_nan(v) or is_inf(v):
			return true
	return false

func _fmt_sci(v: float, digits: int = 4) -> String:
	if v == 0.0:
		return "0"
	var sign := "-" if v < 0.0 else ""
	var av = abs(v)
	var exp := int(floor(log(av) / log(10.0)))
	var mant = av / pow(10.0, exp)
	return sign + String.num(mant, digits) + "e" + str(exp)
