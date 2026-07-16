# Native Windows folder/file picker for the ARES dev server (/pick endpoint).
# Spawned as:  powershell.exe -NoProfile -STA -WindowStyle Hidden -File tools\pick.ps1 -Type folder|file
# -STA is load-bearing: a WinForms dialog on an MTA thread hangs forever (serve.mjs enforces a timeout).
# Writes ONLY the chosen absolute path to stdout (empty line on cancel).
param(
  [ValidateSet('folder','file')] [string]$Type = 'folder',
  [string]$Filter = 'All files (*.*)|*.*',
  [string]$InitialDirectory = ''
)

# PS 5.1 pipes stdout in the console codepage; force UTF-8 so non-ASCII folder names survive to Node.
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

# Invisible TopMost owner so the dialog reliably pops to the foreground (WinForms dialogs have no
# taskbar presence of their own and can otherwise open behind the browser window). Show() is non-blocking.
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = 'CenterScreen'
$owner.Size = New-Object System.Drawing.Size(0, 0)
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()
$owner.Activate()

$selected = $null
try {
  if ($Type -eq 'file') {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = 'Select a file'
    $dlg.Filter = $Filter
    $dlg.Multiselect = $false
    $dlg.CheckFileExists = $true
    if ($InitialDirectory -and (Test-Path $InitialDirectory)) { $dlg.InitialDirectory = $InitialDirectory }
    if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { $selected = $dlg.FileName }
  } else {
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = 'Select the frames folder (OBJ / PLY meshes + atlas PNGs)'
    $dlg.ShowNewFolderButton = $false
    if ($InitialDirectory -and (Test-Path $InitialDirectory)) { $dlg.SelectedPath = $InitialDirectory }
    if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { $selected = $dlg.SelectedPath }
  }
} finally {
  $owner.Close()
  $owner.Dispose()
}

if ($selected) { Write-Output $selected }
