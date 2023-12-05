
add_onnx_mlir_library(OMKrnlToSPIRV
  ConvertKrnlToSPIRV.cpp
  KrnlEntryToGPU.cpp
  KrnlEntryToGPUPass.cpp
  LINK_LIBS PUBLIC
  onnx
)
