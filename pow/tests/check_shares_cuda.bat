@echo off
rem Checks the CUDA training shares against the C reference and the vector files (spec/shares.md section 10), then prints
rem the throughput. Windows, nvcc and MSVC. Set VCVARS and NVCC in the environment if your tools are not in the default
rem install paths.
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul 2>nul
cd /d %~dp0\..\..
if not exist build mkdir build
if not defined NVCC set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
rem ARCH selects the GPU generation (default native, the installed device); dp4a needs sm_61 or later.
if not defined ARCH set "ARCH=native"
set "FAIL=0"
call :one prod "" vectors\shares.txt
call :one s1 "/DNEB_SU1=9 /DNEB_SU2=6" vectors\shares_s1.txt
if "%FAIL%"=="1" (
  echo SHARES CUDA CHECK FAILED
  exit /b 1
)
echo SHARES CUDA CHECK PASSED
build\bench_shares_gpu.exe 3
exit /b 0

:one
rem %1 name of the build, %2 extra C defines (the update shifts of spec/shares.md section 10.1), %3 vector file.
if not exist build\sh_%1 mkdir build\sh_%1
cl /nologo /O2 /TC /W3 /D_CRT_SECURE_NO_WARNINGS %~2 /Ipow\ref /Ipow\tests /c pow\ref\shares.c pow\ref\chacha8.c pow\ref\blake3_min.c pow\tests\syn_dataset.c /Fobuild\sh_%1\ >build\sh_%1.log 2>&1
if errorlevel 1 (
  echo FAIL build C reference %1, see build\sh_%1.log
  set "FAIL=1"
  exit /b 0
)
"%NVCC%" -O3 -arch=%ARCH% -std=c++17 -Ipow\ref -Ipow\gpu -Ipow\tests -o build\test_shares_cuda_%1.exe pow\tests\test_shares_cuda.cu pow\gpu\shares_cuda.cu build\sh_%1\shares.obj build\sh_%1\chacha8.obj build\sh_%1\blake3_min.obj build\sh_%1\syn_dataset.obj >build\sh_%1_nvcc.log 2>&1
if errorlevel 1 (
  echo FAIL build test %1, see build\sh_%1_nvcc.log
  set "FAIL=1"
  exit /b 0
)
if "%1"=="prod" "%NVCC%" -O3 -arch=%ARCH% -std=c++17 -Ipow\ref -Ipow\gpu -Ipow\tests -o build\bench_shares_gpu.exe pow\bench\bench_shares_gpu.cu pow\gpu\shares_cuda.cu build\sh_%1\shares.obj build\sh_%1\chacha8.obj build\sh_%1\blake3_min.obj build\sh_%1\syn_dataset.obj >build\sh_bench_nvcc.log 2>&1
build\test_shares_cuda_%1.exe %3
if errorlevel 1 set "FAIL=1"
exit /b 0
