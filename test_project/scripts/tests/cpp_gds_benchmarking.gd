extends Node
#
#func _ready():
	#run_gdscript_benchmark()
	#run_cpp_benchmark()
#
#func run_gdscript_benchmark():
	#print("✅ GDScript benchmark started")
#
	#var start = Time.get_ticks_usec()
	#var total := 0
#
	#for i in range(1_000_000):
		#total += i % 1000
#
	#var end = Time.get_ticks_usec()
	#var duration_ms = float(end - start) / 1000.0
#
	#print("✅ GDScript benchmark finished, result = ", total)
	#print("⏱ GDScript Time taken: ", duration_ms, " ms")
#
#func run_cpp_benchmark():
	#print("📦 Instantiating C++ Benchmark...")
#
	#var cpp_benchmark = Benchmark.new()  # Must match class name in C++
	#add_child(cpp_benchmark)             # Triggers _ready() in C++
