message(STATUS "include gpt2 directory")

# Source files in this directory
set(GPT2_SOURCE_FILES ${GPT_SOURCE_FILES}
	${CMAKE_CURRENT_LIST_DIR}/IterApprox.cpp
	${CMAKE_CURRENT_LIST_DIR}/MatrixMul.cpp
	${CMAKE_CURRENT_LIST_DIR}/PolyApprox.cpp
	${CMAKE_CURRENT_LIST_DIR}/util.cpp
	${CMAKE_CURRENT_LIST_DIR}/Fold.cpp
	${CMAKE_CURRENT_LIST_DIR}/pack.cpp
	${CMAKE_CURRENT_LIST_DIR}/test_util.cpp
	${CMAKE_CURRENT_LIST_DIR}/optimize.cpp
)

set(GPT2_SOURCE_FILES ${GPT2_SOURCE_FILES} PARENT_SCOPE)
set(GPT2_HEADER_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)