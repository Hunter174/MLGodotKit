extends Node2D

@onready var test_nn_node = $TestNNNode

# Called when the node enters the scene tree for the first time.
func _ready():
	test_nn_node.test()


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
