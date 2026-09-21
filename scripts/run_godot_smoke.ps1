$ErrorActionPreference = "Stop"

$version = if ($env:GODOT_VERSION) { $env:GODOT_VERSION } else { "4.3" }
$release_tag = if ($env:GODOT_RELEASE_TAG) { $env:GODOT_RELEASE_TAG } else { "$version-stable" }
$library_configuration = if ($env:GODOT_LIBRARY_CONFIGURATION) { $env:GODOT_LIBRARY_CONFIGURATION } else { "debug" }
$root = $env:GITHUB_WORKSPACE
$cache = Join-Path $root ".ci/godot-$version"
$archive = Join-Path $env:RUNNER_TEMP "godot-$version.zip"
$download = "https://github.com/godotengine/godot/releases/download/$release_tag/Godot_v$version-stable_win64.exe.zip"

if (-not (Test-Path $cache)) {
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    Invoke-WebRequest -Uri $download -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $cache -Force
}

$godot = Get-ChildItem -Path $cache -Filter "Godot_v$version-stable_win64.exe" -Recurse | Select-Object -First 1
if ($null -eq $godot) {
    throw "Godot executable was not found under $cache"
}

$project = Join-Path $root ".ci/godot-smoke-$version"
if (Test-Path $project) {
    Remove-Item $project -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $project "addons") | Out-Null
$addon = Join-Path $project "addons/mlgodotkit"
Copy-Item -Recurse (Join-Path $root "test_project/addons/mlgodotkit") $addon
if ($library_configuration -eq "release") {
    $manifest = Join-Path $addon "mlgodotkit.gdextension"
    (Get-Content $manifest -Raw).Replace(
        'windows.debug.x86_64 = "res://addons/mlgodotkit/bin/mlgodotkit.windows.template_debug.x86_64.dll"',
        'windows.debug.x86_64 = "res://addons/mlgodotkit/bin/mlgodotkit.windows.template_release.x86_64.dll"') |
        Set-Content $manifest
}

@"
; Generated headless smoke-test project.
config_version=5

[application]
config/name="MLGodotKit Smoke Test"

[display]
window/size/viewport_width=320
window/size/viewport_height=240

[rendering]
renderer/rendering_method="gl_compatibility"
"@ | Set-Content (Join-Path $project "project.godot")

@"
extends SceneTree

func _init():
    var extension = load("res://addons/mlgodotkit/mlgodotkit.gdextension")
    if extension == null:
        quit(1)
        return
    var matrix = ClassDB.instantiate("Matrix")
    if matrix == null:
        quit(1)
        return
    matrix = null
    quit(0)
"@ | Set-Content (Join-Path $project "smoke.gd")

$process = Start-Process -FilePath $godot.FullName `
    -ArgumentList @("--headless", "--path", $project, "--script", "smoke.gd") `
    -Wait -PassThru
$exit_code = $process.ExitCode
if ($exit_code -ne 0) {
    throw "Godot headless smoke test failed with exit code $exit_code"
}

Write-Host "Godot $version headless smoke test passed."
