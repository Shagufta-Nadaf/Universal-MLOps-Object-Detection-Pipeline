@echo off
echo Cleaning old training data...
rd /s /q data\raw      2>nul
rd /s /q data\processed 2>nul
rd /s /q data\splits    2>nul
rd /s /q runs           2>nul
rd /s /q mlruns         2>nul
del /q models\*.pt      2>nul
echo Done! All old data cleaned.
