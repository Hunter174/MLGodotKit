extends ObservableNode


func get_observables():
	# We expose the parents positoon as the target
	return {"target_pos" : parent_node.position}

func reset():
	pass
