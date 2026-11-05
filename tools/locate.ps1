# Folder candidates for a file or folder dropped on the ARES page (serve.mjs /resolve-drop).
# A browser hands a dropped item's NAME, size and modification time, never its location; this
# script lists the places the item most likely came from, and serve.mjs checks each one against
# the name, size and time. Spawned as:
#   powershell.exe -NoProfile -WindowStyle Hidden -File tools\locate.ps1
# The item name arrives in the ARES_DROP_NAME environment variable (never argv, never pasted
# into a command line). Writes one JSON object to stdout:
#   { windows: [folders open in File Explorer], known: [Desktop, Downloads, ...],
#     recent: [targets of Recent\<name>.lnk], index: [full paths from the Windows Search index] }
# Windows PowerShell 5.1 compatible.

$ErrorActionPreference = "SilentlyContinue"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$name = $env:ARES_DROP_NAME
$windows = @(); $known = @(); $recent = @(); $index = @()

# Folders shown in open File Explorer windows: a drag from Explorer starts in one of them.
try {
  $shell = New-Object -ComObject Shell.Application
  foreach ($w in $shell.Windows()) {
    try {
      $p = $w.Document.Folder.Self.Path
      if ($p -and -not $p.StartsWith("::")) { $windows += $p }
    } catch {}
  }
  $dl = $shell.NameSpace("shell:Downloads")
  if ($dl) { $known += $dl.Self.Path }
} catch {}

foreach ($k in @("Desktop", "MyVideos", "MyDocuments", "MyPictures", "CommonDesktopDirectory")) {
  try { $p = [Environment]::GetFolderPath($k); if ($p) { $known += $p } } catch {}
}

if ($name) {
  # Recent items: Windows keeps <name>.lnk for files opened through the shell.
  try {
    $lnk = Join-Path ([Environment]::GetFolderPath("Recent")) ($name + ".lnk")
    if (Test-Path -LiteralPath $lnk) {
      $ws = New-Object -ComObject WScript.Shell
      $t = $ws.CreateShortcut($lnk).TargetPath
      if ($t) { $recent += $t }
    }
  } catch {}

  # Windows Search index: exact file-name match over every indexed location.
  try {
    $q = $name.Replace("'", "''")
    $conn = New-Object -ComObject ADODB.Connection
    $conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows';")
    $rs = $conn.Execute("SELECT TOP 20 System.ItemPathDisplay FROM SYSTEMINDEX WHERE System.FileName = '" + $q + "'")
    while (-not $rs.EOF) { $index += [string]$rs.Fields.Item(0).Value; $rs.MoveNext() }
    $rs.Close(); $conn.Close()
  } catch {}
}

$o = [ordered]@{ windows = @($windows | Select-Object -Unique); known = @($known | Select-Object -Unique); recent = @($recent); index = @($index) }
Write-Output (ConvertTo-Json $o -Compress)
