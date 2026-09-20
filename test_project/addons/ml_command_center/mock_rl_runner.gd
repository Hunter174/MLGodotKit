@tool
extends Node

## MockRLRunner simulates the behavior of a real Reinforcement Learning Runner.
## Use this to test the Command Center UI without needing a full ML backend.

# --- Hyperparameters ---
@export var learning_rate := 0.01:
	set(val):
		learning_rate = val
		print("[MockRLRunner] Learning Rate updated to: ", val)

@export var discount_factor := 0.95:
	set(val):
		discount_factor = val
		print("[MockRLRunner] Discount Factor updated to: ", val)

@export var epsilon := 0.1:
	set(val):
		epsilon = val
		print("[MockRLRunner] Epsilon updated to: ", val)

# --- Telemetry ---
var last_reward := 0.0
var step := 0
var is_training := false

func _process(delta):
	if is_training:
		# Simulate a learning process
		step += 1
		# Randomly fluctuate reward to simulate a learning curve
		last_reward += randf_range(-0.1, 0.11)
		last_reward = clamp(last_reward, -10.0, 10.0)

func start_training() -> void:
	is_training = true
	print("[MockRLRunner] Training Started!")

func stop_training() -> void:
	is_training = false
	print("[MockRLRunner] Training Stopped.")

func reset() -> void:
	step = 0
	last_reward = 0.0
	print("[MockRLRunner] Stats Reset.")
