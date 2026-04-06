param([int]4targetProcessId, [string]4logPath)
while (Get-Process -Id 4targetProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 10
}
Add-Type -AssemblyName System.Windows.Forms
[System.Media.SystemSounds]::Exclamation.Play()
4message = 'Face recognition training finished.'
if (4logPath -and (Test-Path 4logPath)) {
    4tail = Get-Content 4logPath -Tail 12 -ErrorAction SilentlyContinue
    if (4tail) {
        4message = 4message + "

Last log lines:
" + (4tail -join "
")
    }
}
[System.Windows.Forms.MessageBox]::Show(4message, 'Codex Training Complete') | Out-Null
