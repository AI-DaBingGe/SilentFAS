# start_services.ps1
Write-Host "启动多实例 SilentFAS 负载均衡环境..."
Write-Host "正在启动端口 8000 的实例..."
Start-Process -FilePath "python" -ArgumentList "main.py --port 8000" -NoNewWindow
Start-Sleep -Seconds 2

Write-Host "正在启动端口 8001 的实例..."
Start-Process -FilePath "python" -ArgumentList "main.py --port 8001" -NoNewWindow
Start-Sleep -Seconds 2

Write-Host "服务已在后台运行。请确保您已启动 nginx，它会将 80 端口请求负载均衡到 8000 和 8001 端口！"
