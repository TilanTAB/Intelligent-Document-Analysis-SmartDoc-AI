# SmartDoc AI - Virtual Environment Activation Script
# Run this script to activate the virtual environment

Write-Host "?? Activating SmartDoc AI virtual environment..." -ForegroundColor Cyan

# Activate the virtual environment
& ".\venv\Scripts\Activate.ps1"

Write-Host "? Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "?? Installed packages:" -ForegroundColor Yellow
pip list | Select-String -Pattern "langchain|chromadb|gradio|opencv|google-generativeai"
Write-Host ""
Write-Host "?? To run the application:" -ForegroundColor Cyan
Write-Host "   python main.py" -ForegroundColor White
Write-Host ""
Write-Host "?? To deactivate:" -ForegroundColor Cyan
Write-Host "   deactivate" -ForegroundColor White
