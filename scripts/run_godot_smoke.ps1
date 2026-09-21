$ErrorActionPreference = "Stop"

$version = if ($env:GODOT_VERSION) { $env:GODOT_VERSION } else { "4.3" }
$release_tag = if ($env:GODOT_RELEASE_TAG) { $env:GODOT_RELEASE_TAG } else { "$version-stable" }
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

& $godot.FullName --headless --path (Join-Path $root "test_project") --editor --quit
if ($LASTEXITCODE -ne 0) {
    throw "Godot headless smoke test failed with exit code $LASTEXITCODE"
}

Write-Host "Godot $version headless smoke test passed."
