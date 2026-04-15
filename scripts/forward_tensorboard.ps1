param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [string]$RemoteUser,

    [int]$RemotePort = 6006,
    [int]$LocalPort = 16006,
    [string]$IdentityFile = ""
)

if ($RemotePort -le 0) {
    throw "RemotePort must be > 0."
}
if ($LocalPort -le 0) {
    throw "LocalPort must be > 0."
}

$target = "$RemoteUser@$RemoteHost"
$sshArgs = @()
if ($IdentityFile -ne "") {
    $sshArgs += @("-i", $IdentityFile)
}
$sshArgs += @(
    "-N",
    "-L", "$LocalPort`:127.0.0.1`:$RemotePort",
    $target
)

Write-Host "Forwarding TensorBoard port..."
Write-Host "  Remote: $target:$RemotePort"
Write-Host "  Local : http://127.0.0.1:$LocalPort"
Write-Host "Press Ctrl+C to stop forwarding."

ssh @sshArgs

