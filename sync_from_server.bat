@echo off
cd /d "%~dp0"
echo Syncing harvested NBA odds ^& player props from TrueNAS Home Server...
python server_sync.py
pause
