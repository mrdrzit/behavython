$staging = Resolve-Path "build\staging_env"

Write-Output "Starting safe pruning on $staging..."

# 1. Remove .pdb files
$pdbs = Get-ChildItem -Path $staging -Filter "*.pdb" -Recurse -File -ErrorAction SilentlyContinue
$pdbCount = ($pdbs | Measure-Object).Count
if ($pdbs) {
    $pdbs | Remove-Item -Force
}
Write-Output "PDB files removed: $pdbCount"

# 2. Remove include directory
$incPath = Join-Path $staging "include"
$incCount = 0
if (Test-Path $incPath) {
    $incCount = (Get-ChildItem -Path $incPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Remove-Item -Path $incPath -Recurse -Force
}
Write-Output "Include files removed: $incCount"

# 3. Remove Tools directory
$toolsPath = Join-Path $staging "Tools"
$toolsCount = 0
if (Test-Path $toolsPath) {
    $toolsCount = (Get-ChildItem -Path $toolsPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Remove-Item -Path $toolsPath -Recurse -Force
}
Write-Output "Tools files removed: $toolsCount"

# 4. Remove test/tests/testing folders inside site-packages (excluding behavython and deeplabcut)
$sitePackages = Join-Path $staging "Lib\site-packages"
$testDirs = Get-ChildItem -Path $sitePackages -Recurse -Directory -ErrorAction SilentlyContinue | 
    Where-Object { 
        $_.FullName -notmatch '\\behavython\\' -and 
        $_.FullName -notmatch '\\deeplabcut\\' -and 
        ($_.Name -eq 'tests' -or $_.Name -eq 'test' -or $_.Name -eq 'testing')
    }

$testFileCount = 0
foreach ($d in $testDirs) {
    if (Test-Path $d.FullName) {
        $cnt = (Get-ChildItem -Path $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        $testFileCount += $cnt
        Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}
Write-Output "Third-party test files removed: $testFileCount"

$totalPruned = $pdbCount + $incCount + $toolsCount + $testFileCount
Write-Output "Total files pruned: $totalPruned"

$remaining = (Get-ChildItem -Path $staging -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Output "Remaining files to package: $remaining"
