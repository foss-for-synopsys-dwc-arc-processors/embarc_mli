User Tests: Basic API Level Tests 
==============================================
This directory contains basic API level test applications for embARC MLI Library to check 
that all the functions available at the API level work in the way defined by the documentation. 
It is checked with basic error-detecting techniques like thresholds and cyclic redundancy check (CRC32).

# Directory Structure

`/user_tests/make`                 		- contains application specific GNU make rules and settings.  
`/user_tests/test_components`      		- contains sources of various modules which are shared across tests.  
`/user_tests/tests`                		- contains subdirectories with sources and vectors for a test.  


# Building and Running

You need to configure and build the library project for the desired platform. 
Please read the corresponding section on [building the package](/README.md#building-the-package). 
Also take a look at the [User Tests Specific options](#user-tests-specific-extra-options) that you may want to extend configuration with. 
There are no extra requirements specific for this application. All the specified platforms are supported by the test application.  

Build artifacts of the application are stored in the `/obj/<project>/user_tests` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps. 
You need to replace `<options>` placeholder in commands below with the same options list you used for the library configuration and build. 

1. Open command line in the root of the embARC MLI repo and change working directory to './user_tests/make/'

       cd ./user_tests/make/

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build tests. Optional step as you may go to the next step which automatically invokes the build process. 

       gmake <options> build

4. Run all tests

       gmake <options> test_all

Knowing a make target for a specific test, you can run it exclusively, skipping all the rest. 
To get the list of all available test targets use the following command:

       gmake get_tests_list


## User Tests Specific Extra Options. 

There is only a `TEST_DEBUG` pre-processor define which can be passed with external C flags. 
To use it you need to extend your initial [library build command](/README.md#general-build-process) with `EXT_CFLAGS="-DTEST_DEBUG"`:

    gmake <target> <options> EXT_CFLAGS="-DTEST_DEBUG"

This flag unblocks application specific assertions which may help in advanced debugging.


## Expected Output

Test procedure includes running one or multiple (depending on used make target) test groups with generation of report tables. 
Each report table corresponds to a specific test group which is typically run by a separate executable file. 
For more information on the table content see the interface description of reporter modules in [`test_report.h` header file](/user_tests/test_components/test_report.h). 
You can also find more info on the quality metrics in [`test_quality_metrics.h` header file](/user_tests/test_components/test_quality_metrics.h).

Content of the report table may differ depending on the platform and build options. 
In general, the following evidences indicate successful tests passing:

 - `Summary Status: PASSED` in the last line of the test report. 
 - `PASSED` status for each line in the `Result` column of the test report table.
 - `SKIPPED` status together with descriptive message for a test is also an acceptable output. 
 - `0x<CRC32 sum> (OK)` template for each line in the `CRC32 (Status)` column of the test report table. Some tests have only `Result` status and the whole column might be omitted. 

If you see `FAILED` in the `Result` column or `0x<CRC32 sum> (DIFF)` in the `CRC32 (Status)` column 
or `Summary Status: FAILED` in the last line of the table than something is wrong and tests have failed. 

If you use the `test_all` make target, then the first failed test group will halt the whole test procedure. 

# Test Steps of a User Test Application

For most of the tests, input data and expected output data is stored as float values (`*.inc` file inside a specific test directory). 
Test application (`*.cc` file inside a specific test directory) typically does the following:

1. Transforms source data to target data format for a specific test (quantize data).
2. Applies test target function using transformed input operands.
3. Transform kernel output data to float values (dequantize) and compares it against reference float data using various quality metrics. 
4. Calculate CRC from all input and output operands and compares it with hardcoded value to check that results are bit-exact with the expected ones.
5. Form a report table. 

