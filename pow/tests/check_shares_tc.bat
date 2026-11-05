@echo off
rem Differential test of the CUDA kernels K0, K1 and K2 against the C reference, and two negative controls (the tensor-core
rem kernels built wrong on purpose, which the test must reject). Windows, nvcc and MSVC; set VCVARS and NVCC if your tools are
rem not in the default install paths. ARCH is the GPU generation (default native, the installed device); K1 and K2 need sm_80 or later.
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul 2>nul
cd /d %~dp0\..\..
if not exist build mkdir build
if not defined NVCC set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
if not defined ARCH set "ARCH=native"
if not defined KEYS set "KEYS=2048"
rem REAL=dataset file and MODELS=raw model dumps add the real-data cases: set "REALARGS=--real FILE --model F1 --model F2 --real-keys N"
if not defined REALARGS set "REALARGS="
if not exist build\tc_ref mkdir build\tc_ref
cl /nologo /O2 /TC /W3 /D_CRT_SECURE_NO_WARNINGS /Ipow\ref /Ipow\tests /c pow\ref\shares.c pow\ref\chacha8.c pow\ref\blake3_min.c pow\tests\syn_dataset.c /Fobuild\tc_ref\ >build\tc_ref.log 2>&1
if errorlevel 1 ( echo FAIL build C reference, see build\tc_ref.log & exit /b 1 )
set "SRC=pow\tests\test_shares_tc.cu pow\gpu\shares_cuda.cu pow\gpu\shares_tc.cu build\tc_ref\shares.obj build\tc_ref\chacha8.obj build\tc_ref\blake3_min.obj build\tc_ref\syn_dataset.obj"
"%NVCC%" -O3 -arch=%ARCH% -std=c++17 -Xcompiler /openmp -Ipow\ref -Ipow\gpu -Ipow\tests -o build\test_shares_tc.exe %SRC% >build\test_shares_tc_nvcc.log 2>&1
if errorlevel 1 ( echo FAIL build test, see build\test_shares_tc_nvcc.log & exit /b 1 )
build\test_shares_tc.exe %KEYS% %REALARGS%
if errorlevel 1 ( echo SHARES TC CHECK FAILED & exit /b 1 )
"%NVCC%" -O3 -arch=%ARCH% -std=c++17 -Xcompiler /openmp -DNEB_NEGATIVE_CONTROL_TC_CLIP -Ipow\ref -Ipow\gpu -Ipow\tests -o build\test_shares_tc_neg.exe %SRC% >build\test_shares_tc_neg_nvcc.log 2>&1
if errorlevel 1 ( echo FAIL build negative control, see build\test_shares_tc_neg_nvcc.log & exit /b 1 )
build\test_shares_tc_neg.exe 256 >build\test_shares_tc_neg.log 2>&1
if not errorlevel 1 ( echo FAIL the negative control was not detected & exit /b 1 )
echo PASS negative control detected
echo SHARES TC CHECK PASSED
exit /b 0
