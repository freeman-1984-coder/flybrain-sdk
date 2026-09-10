extends Node2D
const Arena = preload("res://arena.gd")
var world = Arena.new()
var http: HTTPRequest
var label: Label
var toggle: Button
var running: bool = false
var busy: bool = false
var session_id: String = ""
var endpoint: String = "http://127.0.0.1:8766"
var operation: String = ""
var issued_at: int = 0
var seq: int = 0
var base_tick: int = 0
var trace: Array = []
var current: Dictionary = {}
var limit: int = 0
var trace_path: String = ""
var last_control: float = 0.0
var failed: bool = false
var restore_path: String = ""
var save_requested: bool = false
var save_path: String = ""
var restoring_world = null
var saving: bool = false

func _ready() -> void:
	http = HTTPRequest.new()
	http.timeout = 2.0
	http.body_size_limit = 16000000
	add_child(http)
	http.request_completed.connect(received)
	label = Label.new()
	label.position = Vector2(30, 20)
	label.size = Vector2(840, 65)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_color_override("font_color", Color("263656"))
	add_child(label)
	toggle = Button.new()
	toggle.text = "Run / pause"
	toggle.position = Vector2(30, 90)
	toggle.pressed.connect(func():
		if busy and not running: return
		running = not running
		if running and not busy: offer()
	)
	add_child(toggle)
	var save_button = Button.new()
	save_button.text = "Save session"
	save_button.position = Vector2(180, 90)
	save_button.pressed.connect(func():
		if failed or session_id.is_empty(): return
		running = false
		save_requested = true
		if not busy: save_session()
	)
	add_child(save_button)
	var restore_button = Button.new()
	restore_button.text = "Restore saved session"
	restore_button.position = Vector2(330, 90)
	restore_button.pressed.connect(func():
		if busy: return
		running = false
		var picker = FileDialog.new()
		picker.access = FileDialog.ACCESS_FILESYSTEM
		picker.current_dir = OS.get_user_data_dir()
		picker.file_mode = FileDialog.FILE_MODE_OPEN_FILE
		picker.add_filter("*.json", "Session JSON")
		picker.file_selected.connect(restore_file)
		add_child(picker)
		picker.popup_centered(Vector2i(700, 450))
	)
	add_child(restore_button)
	var reset_button = Button.new()
	reset_button.text = "Restart"
	reset_button.position = Vector2(550, 90)
	reset_button.pressed.connect(func(): get_tree().reload_current_scene())
	add_child(reset_button)
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--endpoint="): endpoint = arg.trim_prefix("--endpoint=")
		if arg.begins_with("--frames="): limit = int(arg.trim_prefix("--frames="))
		if arg.begins_with("--trace="): trace_path = arg.trim_prefix("--trace=")
		if arg.begins_with("--restore="): restore_path = arg.trim_prefix("--restore=")
	label.text = "Connecting to the local Python CPU brain…"
	if restore_path.is_empty(): send("/start", {})
	else: restore_file(restore_path)

func restore_file(path: String) -> void:
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null or file.get_length() > 16000000:
		stop("Cannot read session (16 MB limit).")
		return
	var data = JSON.parse_string(file.get_as_text())
	if not data is Dictionary or data.get("format") != "flybrain-godot" or data.get("schema_version") != 1 or not data.get("environment") is Dictionary or not data.get("controller_json") is String:
		stop("Invalid session format.")
		return
	restoring_world = Arena.from_snapshot(data.environment)
	if restoring_world == null or data.get("next_seq") != restoring_world.frame:
		stop("Invalid world or mismatched session clocks.")
		return
	failed = false
	toggle.disabled = false
	trace.clear()
	send("/start", {"checkpoint_json": data.controller_json})

func save_session() -> void:
	save_requested = false
	saving = true
	save_path = "user://flybrain-session-%d.json" % Time.get_ticks_usec()
	send("/checkpoint", {})

func send(path: String, body: Dictionary) -> void:
	if busy: return
	busy = true
	operation = path
	issued_at = Time.get_ticks_msec()
	if path != "/start": body.session = session_id
	var error = http.request(endpoint + path, ["Content-Type: application/json"], HTTPClient.METHOD_POST, JSON.stringify(body, "", true, true))
	if error != OK: stop("Cannot connect. Start bridge.py and restart the scene.")

func offer() -> void:
	if not running or busy or failed: return
	current = {"observation": world.observe()}
	send("/offer", {"seq": seq, "observation": current.observation})

func valid_integer(value, maximum: float = 9007199254740991.0) -> bool:
	return (value is int or value is float) and is_finite(value) and value >= 0 and value <= maximum and floor(value) == value

func received(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	busy = false
	if failed: return
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		stop("Bridge rejected/unavailable: " + body.get_string_from_utf8().left(250) + " Restart both participants.")
		return
	var value = JSON.parse_string(body.get_string_from_utf8())
	if not value is Dictionary:
		stop("Invalid bridge response.")
		return
	if operation == "/start":
		if not value.get("session") is String or value.session.is_empty() or value.session.length() > 128 or not value.get("model") is String or not valid_integer(value.get("next_seq"), 4294967295.0) or not valid_integer(value.get("base_tick")):
			stop("Invalid start response.")
			return
		session_id = value.session
		seq = int(value.next_seq)
		base_tick = int(value.base_tick)
		var expected_frame = restoring_world.frame if restoring_world != null else world.frame
		if value.next_seq != expected_frame or value.period_ms != 20:
			stop("Restored controller and world clocks differ.")
			return
		if restoring_world != null:
			world = restoring_world
			restoring_world = null
			queue_redraw()
		label.text = "No CUDA required · " + str(value.model) + "\nEngine-owned scene · assumed LIF + engineered steering"
		running = limit > 0
		if running: offer()
	elif operation == "/offer":
		if not valid_integer(value.get("seq")) or not valid_integer(value.get("brain_tick")) or value.get("seq") != seq or value.get("duration_ms") != 20.0 or value.get("brain_tick") != base_tick + (seq + 1) * 20 or Time.get_ticks_msec() - issued_at > 2000:
			stop("Outdated action rejected. Restart both participants.")
			return
		if not value.get("requested") is Dictionary or value.requested.keys() != ["steer"] or not (value.requested.steer is float or value.requested.steer is int) or not is_finite(value.requested.steer):
			stop("Invalid steering response.")
			return
		current.response = value
		current.applied = world.apply(value.requested)
		current.environment = world.snapshot()
		last_control = current.applied.steer
		queue_redraw()
		send("/ack", {"seq": seq, "applied": current.applied})
	elif operation == "/ack":
		if not valid_integer(value.get("ack")) or value.get("ack") != seq:
			stop("Acknowledgement mismatch.")
			return
		trace.append(current.duplicate(true))
		seq += 1
		if save_requested:
			save_session()
			return
		if limit > 0 and seq >= limit:
			running = false
			send("/checkpoint", {})
			return
		await get_tree().create_timer(0.02).timeout
		if running: offer()
	elif operation == "/checkpoint":
		if not value.get("checkpoint_json") is String or value.checkpoint_json.is_empty():
			stop("Invalid checkpoint response.")
			return
		var destination = save_path if saving else trace_path
		if not destination.is_empty():
			var file = FileAccess.open(destination, FileAccess.WRITE)
			if file == null:
				stop("Cannot write trace.")
				return
			file.store_string(JSON.stringify({"format": "flybrain-godot", "schema_version": 1, "frames": trace, "controller_json": value.checkpoint_json, "next_seq": seq, "environment": world.snapshot()}, "", true, true))
		if saving:
			label.text = "Session saved: " + ProjectSettings.globalize_path(save_path)
			saving = false
			return
		print("GODOT_OK frames=", seq, " steer=", last_control)
		get_tree().quit(0)

func stop(message: String) -> void:
	failed = true
	running = false
	busy = false
	last_control = 0.0
	toggle.disabled = true
	label.text = message
	queue_redraw()
	push_error(message)
	if limit > 0:
		if not trace_path.is_empty():
			var file = FileAccess.open(trace_path, FileAccess.WRITE)
			if file != null:
				file.store_string(JSON.stringify({"failed": true, "environment": world.snapshot(), "applied": last_control}))
		get_tree().quit(1)

func _draw() -> void:
	draw_rect(Rect2(30, 150, 840, 430), Color("e7edf8"))
	for b in world.blocks:
		draw_rect(Rect2(50 + b.x * 800 - 12, 170 + b.y * 370 - 8, 24, 16), Color("de996c"))
	draw_circle(Vector2(50 + world.x * 800, 540), 14, Color("4263df"))
	if label:
		draw_string(ThemeDB.fallback_font, Vector2(30, 615), "frame %d · passed %d · collisions %d · applied %.3f" % [world.frame, world.passed, world.collisions, last_control], HORIZONTAL_ALIGNMENT_LEFT, -1, 18, Color("263656"))
