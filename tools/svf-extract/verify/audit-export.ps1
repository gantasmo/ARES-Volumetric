# Full data-only audit of the SVF export. No imagery is opened — this reads OBJ text (coordinates)
# and the manifest only.
#
# The point: verify the vertex LAYOUT independently of the plugin's own bounds. If our stride/offset
# reading were wrong, UVs would not land in [0,1] and normals would not be unit length. Those two
# checks can't pass by luck.
$d = "D:\SVF-export"
$rows = Get-Content "$d\manifest.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
$bad = 0; $n = 0
$prevCentroid = $null; $identical = 0; $centroidChecks = 0
$minMove = [double]::MaxValue; $maxMove = [double]::MinValue
$uvMin = [double]::MaxValue; $uvMax = [double]::MinValue
$nlenMin = [double]::MaxValue; $nlenMax = [double]::MinValue
$faceIdxBad = 0

foreach ($m in $rows) {
  $f = Join-Path $d ("mesh-f{0:d5}.obj" -f ($m.frame + 1))
  if (-not (Test-Path $f)) { "frame $($m.frame): OBJ MISSING"; $bad++; continue }
  $v = 0; $vt = 0; $vn = 0; $face = 0; $nz = 0
  $sx = 0.0; $sy = 0.0; $sz = 0.0
  $inb = 0
  $b = $m.bounds
  foreach ($line in [System.IO.File]::ReadLines($f)) {
    $c0 = $line[0]
    if ($c0 -eq 'v' -and $line[1] -eq ' ') {
      $p = $line.Split(' '); $x = [double]$p[1]; $y = [double]$p[2]; $z = [double]$p[3]
      $v++; $sx += $x; $sy += $y; $sz += $z
      if ($x -ne 0 -or $y -ne 0 -or $z -ne 0) { $nz++ }
      if ($x -ge $b[0] - 1 -and $x -le $b[3] + 1 -and $y -ge $b[1] - 1 -and $y -le $b[4] + 1 -and $z -ge $b[2] - 1 -and $z -le $b[5] + 1) { $inb++ }
    }
    elseif ($c0 -eq 'v' -and $line[1] -eq 't') {
      $vt++
      if ($vt -le 400) {
        $p = $line.Split(' '); $u = [double]$p[1]; $w = [double]$p[2]
        foreach ($q in @($u, $w)) { if ($q -lt $uvMin) { $uvMin = $q }; if ($q -gt $uvMax) { $uvMax = $q } }
      }
    }
    elseif ($c0 -eq 'v' -and $line[1] -eq 'n') {
      $vn++
      if ($vn -le 400) {
        $p = $line.Split(' ')
        $len = [math]::Sqrt(([double]$p[1]) * ([double]$p[1]) + ([double]$p[2]) * ([double]$p[2]) + ([double]$p[3]) * ([double]$p[3]))
        if ($len -lt $nlenMin) { $nlenMin = $len }; if ($len -gt $nlenMax) { $nlenMax = $len }
      }
    }
    elseif ($c0 -eq 'f') {
      $face++
      if ($face -le 200) {
        foreach ($tok in $line.Split(' ')[1..3]) {
          $vi = [int]($tok.Split('/')[0])
          if ($vi -lt 1 -or $vi -gt $m.verts) { $faceIdxBad++ }
        }
      }
    }
  }
  $n++
  $vOk = ($v -eq $m.verts); $fOk = ($face -eq [math]::Floor($m.indices / 3))
  $allNz = ($nz -eq $v); $allIn = ($inb -eq $v)
  if (-not ($vOk -and $fOk -and $allNz -and $allIn -and $vt -eq $v -and $vn -eq $v)) {
    $bad++
    "frame {0}: v={1}/{2} f={3} nz={4}/{5} inBounds={6}/{7} vt={8} vn={9}  <-- PROBLEM" -f $m.frame, $v, $m.verts, $face, $nz, $v, $inb, $v, $vt, $vn
  }
  # Parenthesise each element: in PowerShell `@($a / $b, $c / $b)` parses the commas as array
  # construction and divides BY an array, which throws — leaving the check a silent no-op.
  $cent = @(($sx / $v), ($sy / $v), ($sz / $v))
  if ($null -ne $prevCentroid) {
    $dd = [math]::Sqrt((($cent[0] - $prevCentroid[0]) * ($cent[0] - $prevCentroid[0])) + (($cent[1] - $prevCentroid[1]) * ($cent[1] - $prevCentroid[1])) + (($cent[2] - $prevCentroid[2]) * ($cent[2] - $prevCentroid[2])))
    $centroidChecks++
    if ($dd -lt 1e-6) { $identical++ }
    if ($dd -lt $minMove) { $minMove = $dd }
    if ($dd -gt $maxMove) { $maxMove = $dd }
  }
  $prevCentroid = $cent
}

""
"================ EXPORT AUDIT ================"
"frames audited        : $n"
"frames with problems  : $bad"
"centroid comparisons  : $centroidChecks  (must equal frames-1, else the check silently no-op'd)"
"identical-to-previous : $identical  (0 = every frame is genuinely different geometry)"
"centroid motion/frame : [{0:0.000}, {1:0.000}] mm" -f $minMove, $maxMove
"UV range (sampled)    : [{0:0.0000}, {1:0.0000}]   expect ~[0,1] -> validates the uv offset" -f $uvMin, $uvMax
"normal length (sampled): [{0:0.0000}, {1:0.0000}]  expect ~1.0    -> validates the normal offset" -f $nlenMin, $nlenMax
"face indices out of range (sampled): $faceIdxBad"
"=============================================="
