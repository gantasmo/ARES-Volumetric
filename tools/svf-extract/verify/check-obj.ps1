# Data-only validation of an exported frame: does the OBJ actually contain the geometry the plugin
# said it would? Counts + coordinate ranges vs the manifest's own reported bounds. No imagery.
$d = "D:\SVF-export"
$manifest = Get-Content "$d\manifest.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
foreach ($m in $manifest) {
  $f = Join-Path $d ("mesh-f{0:d5}.obj" -f ($m.frame + 1))
  if (-not (Test-Path $f)) { "frame $($m.frame): OBJ MISSING ($f)"; continue }
  $v = 0; $vt = 0; $vn = 0; $face = 0
  $xmn = [double]::MaxValue; $xmx = [double]::MinValue
  $ymn = [double]::MaxValue; $ymx = [double]::MinValue
  $zmn = [double]::MaxValue; $zmx = [double]::MinValue
  $nz = 0
  foreach ($line in [System.IO.File]::ReadLines($f)) {
    if ($line.StartsWith("v ")) {
      $v++
      $p = $line.Split(' ')
      $x = [double]$p[1]; $y = [double]$p[2]; $z = [double]$p[3]
      if ($x -ne 0 -or $y -ne 0 -or $z -ne 0) { $nz++ }
      if ($x -lt $xmn) { $xmn = $x }; if ($x -gt $xmx) { $xmx = $x }
      if ($y -lt $ymn) { $ymn = $y }; if ($y -gt $ymx) { $ymx = $y }
      if ($z -lt $zmn) { $zmn = $z }; if ($z -gt $zmx) { $zmx = $z }
    }
    elseif ($line.StartsWith("vt ")) { $vt++ }
    elseif ($line.StartsWith("vn ")) { $vn++ }
    elseif ($line.StartsWith("f ")) { $face++ }
  }
  $b = $m.bounds
  # Does every axis range sit inside the plugin's own reported AABB (5% slop)?
  $ok = ($xmn -ge $b[0] - 1) -and ($xmx -le $b[3] + 1) -and ($ymn -ge $b[1] - 1) -and ($ymx -le $b[4] + 1) -and ($zmn -ge $b[2] - 1) -and ($zmx -le $b[5] + 1)
  $vOk = ($v -eq $m.verts)
  $fOk = ($face -eq [math]::Floor($m.indices / 3))
  ""
  "frame $($m.frame)  ($([System.IO.Path]::GetFileName($f)))"
  "  v={0} (manifest {1}) {2}   vt={3} vn={4} f={5} (expect {6}) {7}" -f $v, $m.verts, $(if ($vOk) { "OK" } else { "MISMATCH" }), $vt, $vn, $face, [math]::Floor($m.indices / 3), $(if ($fOk) { "OK" } else { "MISMATCH" })
  "  non-zero verts : {0}/{1}  ({2:0.0}%)" -f $nz, $v, (100 * $nz / [math]::Max(1, $v))
  "  x[{0:0.00},{1:0.00}] y[{2:0.00},{3:0.00}] z[{4:0.00},{5:0.00}]" -f $xmn, $xmx, $ymn, $ymx, $zmn, $zmx
  "  manifest AABB  x[{0:0.00},{1:0.00}] y[{2:0.00},{3:0.00}] z[{4:0.00},{5:0.00}]" -f $b[0], $b[3], $b[1], $b[4], $b[2], $b[5]
  "  inside bounds  : {0}" -f $(if ($ok) { "YES" } else { "NO" })
}
