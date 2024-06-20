cmake_minimum_required(VERSION 3.16)

project(gpt2_ckks_project)
set(CMAKE_CXX_COMPILER g++)

option(COMP "Build comparison operation" OFF)
option(BOOT "Build boot operation" OFF)
option(CNN "Build comparison operation" OFF)
option(APPROX_TEST "Build comparison operation" OFF)
option(LAYER_TEST "Build comparison operation" OFF)
option(MICROBENCH "Build microbenchmark configuration" OFF)
option(BENCH "Build benhcmark configuration" OFF)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./run)
include("~/tmleong/mylibs/lib/cmake/SEAL-3.6/SEALConfig.cmake")
include("~/tmleong/mylibs/lib/cmake/hexl-1.2.0/HEXLConfig.cmake")
#find_package(SEAL 3.6 REQUIRED)

add_subdirectory(common)
add_subdirectory(gpt2-ckks/single-key/gpt2)
add_subdirectory(gpt2-ckks/single-key/ckks_bootstrapping)


if(COMP)
	add_subdirectory(common)
	add_subdirectory(cpu-ckks/single-key/comp)
	add_executable(compare)
	target_sources(compare
		PRIVATE
			./run/run_compare.cpp
			${COMMON_SOURCE_FILES}
			${COMP_SOURCE_FILES}
	)
	target_compile_options(compare PRIVATE -Wall -O3 -std=c++11 -pthread -march=native)
	target_link_libraries(compare PRIVATE ntl gmp m pthread SEAL::seal)
	target_include_directories(compare PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${COMP_HEADER_DIR})
endif()

if(BOOT)
	message(STATUS "Inside BOOT")
	link_directories("/home/ubuntu/tmleong/mylibs/lib")
	add_executable(bootstrapping)
	target_include_directories(bootstrapping PRIVATE ~/tmleong/mylibs/include)
	target_link_libraries(bootstrapping PRIVATE omp)
	target_sources(bootstrapping
		PRIVATE
			./run/run_bootstrapping.cpp
			${COMMON_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(bootstrapping PRIVATE -Wall -O0 -std=c++17 -fopenmp -pthread -march=native)
	target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread SEAL::seal)
	target_include_directories(bootstrapping PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR} )

	add_executable(bootstrap_basic)
	target_include_directories(bootstrap_basic PRIVATE ~/tmleong/mylibs/include)
	target_link_libraries(bootstrap_basic PRIVATE omp)
	target_sources(bootstrap_basic
		PRIVATE
			./run/run_minimal_bootstrap.cpp
			${COMMON_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
	)

	target_compile_options(bootstrap_basic PRIVATE -Wall -O0 -std=c++17 -fopenmp -pthread -march=native)
	target_link_libraries(bootstrap_basic PRIVATE ntl gmp m pthread SEAL::seal)
	target_include_directories(bootstrap_basic PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR} ${GPT2_HEADER_DIR})
endif()


if(APPROX_TEST)
	message(STATUS "Inside TEST")
	link_directories("/home/ubuntu/tmleong/mylibs/lib")
	add_executable(approx_test)
	target_include_directories(approx_test PRIVATE ~/tmleong/mylibs/include)
	target_link_libraries(approx_test PRIVATE omp)
	target_sources(approx_test
		PRIVATE
			./run/run_approx_test.cpp
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(approx_test PRIVATE -w -O0 -std=c++17 -fopenmp -pthread -march=native)
	target_link_libraries(approx_test PRIVATE ntl gmp pthread gomp SEAL::seal)
	target_include_directories(approx_test PUBLIC ~/tmleong/mylibs/include ~/tmleong/mylibs/include/NTL /usr/include  /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})

endif()

if(MICROBENCH)
message(STATUS "Inside MICROBENCH")
	add_executable(microbench)
	target_sources(microbench
		PRIVATE
			./run/microbenchmark.cpp
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(microbench PRIVATE -w -O0 -std=c++17 -pthread -march=native -fopenmp)
	target_link_libraries(microbench PRIVATE ntl gmp pthread SEAL::seal gomp protobuf)
	target_include_directories(microbench PUBLIC ~/tmleong/FHE-GPT-2/gpt2_weights ~/tmleong/mylibs/include /usr/include/ /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
endif()

if(BENCH)
message(STATUS "Inside BENCH")
	add_executable(bench)
	target_sources(bench
		PRIVATE
			./run/infer_gpt2.cpp
			/home/tmleong/FHE-GPT-2/gpt2_weights/weights.pb.cc
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)

	find_package(OpenMP REQUIRED)

	target_compile_options(bench PRIVATE -w -O3 -std=c++17 -pthread -march=native -fopenmp)
	target_link_libraries(bench PRIVATE ntl gmp pthread SEAL::seal gomp protobuf)
	target_include_directories(bench PUBLIC ~/tmleong/FHE-GPT-2/gpt2_weights /usr/include/ /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
endif()