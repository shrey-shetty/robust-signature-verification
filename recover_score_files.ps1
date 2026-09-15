# Run from the repo root:
#   C:\Users\Shreya\Downloads\robust-signature-verification
# Restores test_scores.csv for resnet34, rq3_combined, and (probably) efficientnet_b0
# into the results\final\<arm>\<dataset>\ convention.

# --- 1. check whether results_rq2.zip is the efficientnet arm ----------------
Expand-Archive -Path "$HOME\Downloads\results_rq2.zip" -DestinationPath .\results\rq2_unzip -Force
Get-ChildItem -Recurse -Path .\results\rq2_unzip -Filter test_scores.csv |
  Select-Object -ExpandProperty FullName
Get-ChildItem -Recurse -Path .\results\rq2_unzip -Filter config.json |
  ForEach-Object { Write-Host "`n--- $($_.FullName)"; (Get-Content $_.FullName | ConvertFrom-Json).backbone }

# STOP here and read the output. The config.json backbone field tells you which
# arm this zip actually holds. Only continue if it says efficientnet_b0.

# --- 2. place resnet34 (institutional only) ---------------------------------
$dest = ".\results\final\resnet34\institutional"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item ".\results\resnet34_unzip\results_rq2_resnet34\eval_resnet34\*" -Destination $dest

# --- 3. place rq3_combined (all four corpora) -------------------------------
foreach ($d in 'institutional','cedar','bhsig260_bengali','bhsig260_hindi') {
  $dest = ".\results\final\rq3_combined\$d"
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  Copy-Item ".\results\rq3_unzip\results_rq3\eval_rq3_$d\*" -Destination $dest
}

# --- 4. place efficientnet_b0 -- ONLY after step 1 confirms the arm ---------
# Adjust the source path to match what step 1 actually printed.
# foreach ($d in 'institutional','cedar','bhsig260_bengali','bhsig260_hindi') {
#   $dest = ".\results\final\efficientnet_b0\$d"
#   New-Item -ItemType Directory -Force -Path $dest | Out-Null
#   Copy-Item ".\results\rq2_unzip\<actual-folder>\eval_efficientnet_b0_$d\*" -Destination $dest
# }

# --- 5. final inventory -----------------------------------------------------
Get-ChildItem -Recurse -Path .\results\final -Filter test_scores.csv |
  Select-Object -ExpandProperty FullName
