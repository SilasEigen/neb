@echo off
rem Checks the nebhash v2 CUDA scan against the C reference, then prints the throughput. Windows, nvcc and MSVC. Set VCVARS
rem and NVCC in the environment if your tools are not in the default install paths; ARCH selects the GPU generation
rem (default native, the installed device).
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul 2>nul
cd /d %~dp0\..\..
if not exist build mkdir build
if not defined NVCC set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
if not defined ARCH set "ARCH=native"
"%NVCC%" -O3 -arch=%ARCH% -std=c++17 -Ipow\ref -Ipow\gpu -Ipow\tests -o build\test_nebhash2_cuda.exe pow\tests\test_nebhash2_cuda.cu pow\gpu\nebhash2_cuda.cu pow\ref\nebhash2.c pow\ref\shares.c pow\ref\chacha8.c pow\ref\blake3_min.c pow\tests\syn2_dataset.c >build\nebhash2_cuda_nvcc.log 2>&1
if errorlevel 1 (
  echo FAIL build, see build\nebhash2_cuda_nvcc.log
  exit /b 1
)
build\test_nebhash2_cuda.exe %*
