extends Node2D

func _ready():
	var A = [[1, 2], [3, 4]]
	var B = [[5, 6], [7, 8]]
	var C = Linalg.add(A, B)
	print(C)
