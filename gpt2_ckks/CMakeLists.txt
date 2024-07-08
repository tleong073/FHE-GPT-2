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
option(SANITY_CHECK "Build benchmark configuration" OFF)
option(OMP_TEST "Build benhcmark configuration" OFF)
option(SECURE_INFER "Secure embedding" OFF)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./run)
#include("~/tmleong/mylibs/lib/cmake/SEAL-3.6/SEALConfig.cmake")
#find_package(SEAL 3.6 REQUIRED)
find_package(SEAL 4.1 REQUIRED)
find_package(HEXL 1.2.5 REQUIRED)
#find_package(SEAL 3.6 REQUIRED)

add_subdirectory(common)
add_subdirectory(gpt2-ckks/single-key/gpt2)
add_subdirectory(gpt2-ckks/single-key/ckks_bootstrapping)

if(SANITY_CHECK)
	message(STATUS "Inside TEST")
	add_executable(sanity_check)
	target_sources(sanity_check
		PRIVATE
			./run/4_ckks_basics.cpp
			./run/examples.cpp
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(sanity_check PRIVATE -w -O0 -std=c++17 -fopenmp -pthread -march=native)
	target_link_libraries(sanity_check PRIVATE ntl gmp pthread gomp SEAL::seal)
	target_include_directories(sanity_check PUBLIC run/ /usr/include  /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
endif()
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
	add_executable(bootstrapping)
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
	target_sources(bootstrap_basic
		PRIVATE
			./run/run_minimal_bootstrap.cpp
			${COMMON_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
	)

	target_compile_options(bootstrap_basic PRIVATE -Wall -O0 -std=c++17 -fopenmp -pthread -march=native)
	target_link_libraries(bootstrap_basic PRIVATE ntl gmp m pthread HEXL::hexl SEAL::seal)
	target_include_directories(bootstrap_basic PUBLIC /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR} ${GPT2_HEADER_DIR})
endif()


if(APPROX_TEST)
	message(STATUS "Inside TEST")
	add_executable(approx_test)
	target_sources(approx_test
		PRIVATE
			./run/run_approx_test.cpp
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(approx_test PRIVATE -w -O0 -std=c++17 -fopenmp -pthread)
	target_link_libraries(approx_test PRIVATE ntl gmp pthread gomp HEXL::hexl SEAL::seal)
	target_include_directories(approx_test PUBLIC /usr/include  /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})

endif()

if(OMP_TEST)
	message(STATUS "Inside OMP TEST")
	add_executable(omp_test)
	target_sources(omp_test
		PRIVATE
			./run/run_omp_test.cpp
			${COMMON_SOURCE_FILES}
			${GPT2_SOURCE_FILES}
			${COMP_SOURCE_FILES}
			${BOOTSTRAPPING_SOURCE_FILES}
	)
	target_compile_options(omp_test PRIVATE -w -O2 -std=c++17 -fopenmp -pthread)
	target_link_libraries(omp_test PRIVATE ntl gmp pthread gomp HEXL::hexl SEAL::seal)
	target_include_directories(omp_test PUBLIC /usr/include  /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
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
	target_compile_options(microbench PRIVATE -w -O2 -std=c++17 -pthread -march=native -fopenmp)
	target_link_libraries(microbench PRIVATE ntl gmp pthread HEXL::hexl SEAL::seal gomp protobuf)
	target_include_directories(microbench PUBLIC /usr/include/ /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
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

include(network/cmake/common.cmake)

# Proto file
get_filename_component(si_proto "protos/secure_inference.proto" ABSOLUTE)
get_filename_component(si_proto_path "${si_proto}" PATH)

# Generated sources
set(si_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/secure_inference.pb.cc")
set(si_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/secure_inference.pb.h")
set(si_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/secure_inference.grpc.pb.cc")
set(si_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/secure_inference.grpc.pb.h")
add_custom_command(
      OUTPUT "${si_proto_srcs}" "${si_proto_hdrs}" "${si_grpc_srcs}" "${si_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${si_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${si_proto}"
      DEPENDS "${si_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# rg_grpc_proto
add_library(si_grpc_proto
  ${si_grpc_srcs}
  ${si_grpc_hdrs}
  ${si_proto_srcs}
  ${si_proto_hdrs})
target_link_libraries(si_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})



if(SECURE_INFER)
	message(STATUS "Inside secure_embed")

	foreach(_target
		secure_inference_client secure_inference_server)
			add_executable(${_target}
			"network/${_target}.cc")
			target_link_libraries(${_target}
			si_grpc_proto
			SEAL::seal
			${_REFLECTION}
			${_GRPC_GRPCPP}
			${_PROTOBUF_LIBPROTOBUF})
	endforeach()

	# add_executable(sanity_check)
	# target_sources(sanity_check
	# 	PRIVATE
	# 		./run/4_ckks_basics.cpp
	# 		./run/examples.cpp
	# 		${COMMON_SOURCE_FILES}
	# 		${GPT2_SOURCE_FILES}
	# 		${COMP_SOURCE_FILES}
	# 		${BOOTSTRAPPING_SOURCE_FILES}
	# )
	# target_compile_options(sanity_check PRIVATE -w -O0 -std=c++17 -fopenmp -pthread -march=native)
	# target_link_libraries(sanity_check PRIVATE ntl gmp pthread gomp SEAL::seal)
	# target_include_directories(sanity_check PUBLIC run/ /usr/include  /usr/local/include ${COMMON_HEADER_DIR} ${GPT2_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
endif()