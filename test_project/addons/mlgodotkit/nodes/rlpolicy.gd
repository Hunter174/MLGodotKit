extends Node
class_name RLPolicy

@onready var nn: NNNode = $"../PolicyNN"


func act(state: Array) -> Array:
	## Forward through NN to choose an action (currently deterministic)
	return nn.forward(state)

func evaluate(state: Array, action: Array) -> Dictionary:
	## Optional: compute log_prob or value if needed (stub for PPO)
	return {"log_prob": 0.0, "value": 0.0}
