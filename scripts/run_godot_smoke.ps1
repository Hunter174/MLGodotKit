$ErrorActionPreference = "Stop"

$version = if ($env:GODOT_VERSION) { $env:GODOT_VERSION } else { "4.3" }
$release_tag = if ($env:GODOT_RELEASE_TAG) { $env:GODOT_RELEASE_TAG } else { "$version-stable" }
$library_configuration = if ($env:GODOT_LIBRARY_CONFIGURATION) { $env:GODOT_LIBRARY_CONFIGURATION } else { "debug" }
$root = if ($env:GITHUB_WORKSPACE) { $env:GITHUB_WORKSPACE } else { (Resolve-Path ".").Path }
$temp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { Join-Path $root ".ci/tmp" }
$cache = Join-Path $root ".ci/godot-$version"
$archive = Join-Path $temp "godot-$version.zip"
$godot_path = Join-Path $cache "Godot_v$version-stable_win64.exe"
$download = "https://github.com/godotengine/godot/releases/download/$release_tag/Godot_v$version-stable_win64.exe.zip"

if (-not (Test-Path $godot_path)) {
    if (Test-Path $cache) {
        Remove-Item $cache -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    Invoke-WebRequest -Uri $download -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $cache -Force
}

$godot = Get-Item $godot_path

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

[editor_plugins]
enabled=PackedStringArray("res://addons/mlgodotkit/plugins/plugin.cfg")
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

$stdout = Join-Path $temp "godot-editor-$version.stdout.log"
$stderr = Join-Path $temp "godot-editor-$version.stderr.log"
$editor_process = Start-Process -FilePath $godot.FullName `
    -ArgumentList @("--headless", "--editor", "--path", $project, "--quit-after", "5") `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -Wait -PassThru
$editor_output = ((Get-Content $stdout -Raw), (Get-Content $stderr -Raw)) -join "`n"
Write-Host $editor_output
if ($editor_process.ExitCode -ne 0) {
    throw "Godot editor smoke test failed with exit code $($editor_process.ExitCode)"
}
if ($editor_output -match "SCRIPT ERROR|Failed to load|ERROR:") {
    throw "Godot editor smoke test reported script or resource errors"
}

Write-Host "Godot $version runtime and editor smoke tests passed."
