# Install Eigen for Windows CI
$ErrorActionPreference = "Stop"

$workspace = $env:GITHUB_WORKSPACE
$zipFile = "eigen.zip"
$tmpDir = "$workspace/eigen_tmp"
$destDir = "$workspace/eigen"

Write-Host "Downloading Eigen 3.4.0..."
Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip' -OutFile $zipFile

Write-Host "Extracting Eigen..."
if (Test-Path $tmpDir) { Remove-Item -Path $tmpDir -Recurse -Force }
Expand-Archive -Path $zipFile -DestinationPath $tmpDir -Force

Write-Host "Moving to final destination..."
if (Test-Path $destDir) { Remove-Item -Path $destDir -Recurse -Force }
$extractedFolder = Get-ChildItem -Path $tmpDir -Directory | Select-Object -First 1
Move-Item -Path $extractedFolder.FullName -Destination $destDir -Force

Write-Host "Cleaning up..."
Remove-Item -Path $zipFile -Force
Remove-Item -Path $tmpDir -Recurse -Force

Write-Host "Eigen successfully installed at $destDir"
Get-ChildItem -Path $destDir
