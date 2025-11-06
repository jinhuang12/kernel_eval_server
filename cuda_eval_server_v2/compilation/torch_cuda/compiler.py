"""
CuPy compiler for TORCH_CUDA kernels
Compiles extracted CUDA kernels using CuPy with C++ wrapper transformation
"""

import os
import logging
import cupy
from typing import Dict, Any, Optional, List
import re
from torch.utils.cpp_extension import load_inline

from .kernel_extractor import ExtractedKernel
from .cpp_wrapper_transformer import TorchCudaCppWrapperTransformer


logger = logging.getLogger(__name__)


class TorchCudaCupyCompiler:
    """
    Compiles extracted CUDA kernels using CuPy
    Handles C++ wrapper transformation for execution
    """
    
    def __init__(self, cache_dir: str = "/tmp/cupy_kernel_cache"):
        self.cache_dir = cache_dir
        self.cpp_wrapper = TorchCudaCppWrapperTransformer()

    def compile_cuda_kernel(self, kernel: ExtractedKernel, gpu_id: int) -> Dict[str, Any]:
        """
        Compile extracted CUDA kernel with CuPy
        
        Args:
            kernel: ExtractedKernel with CUDA source and metadata
            gpu_id: GPU device ID for compilation
            
        Returns:
            Dictionary with compilation results
        """
        try:
            with cupy.cuda.Device(gpu_id):
                logger.info(f"Compiling kernel {kernel.name} on GPU {gpu_id}")
                
                # Preprocess CUDA source for CuPy
                preprocessed_cuda = self._preprocess_cuda_source(kernel.cuda_source)
                
                # Compile CUDA kernels with CuPy
                compiled_functions = {}
                compilation_errors = []
                
                for func_name in kernel.kernel_functions:
                    try:
                        # Compile each kernel function
                        raw_kernel = cupy.RawKernel(
                            preprocessed_cuda,
                            func_name
                        )
                        raw_kernel.compile()  # Compiles on the current device (gpu_id)
                        compiled_functions[func_name] = raw_kernel
                        logger.info(f"  Compiled kernel function: {func_name}")
                    except Exception as e:
                        error_msg = f"Failed to compile {func_name}: {e}"
                        logger.error(error_msg)
                        compilation_errors.append(error_msg)
                
                if not compiled_functions and compilation_errors:
                    return {
                        "compilation_successful": False,
                        "compilation_errors": compilation_errors,
                        "kernel_name": kernel.name
                    }
                
                # Transform and compile C++ wrapper (MUST succeed)
                cpp_wrapper_result = self._compile_cpp_wrapper(kernel)
                if not cpp_wrapper_result or not cpp_wrapper_result.get("compilation_successful", False):
                    logger.error("C++ wrapper compilation failed - this is required")
                    return {
                        "compilation_successful": False,
                        "error": f"C++ wrapper compilation failed: {cpp_wrapper_result.get('error') if cpp_wrapper_result else 'Unknown error'}",
                        "kernel_name": kernel.name
                    }
                
                # Prepare compilation result
                result = {
                    "compilation_successful": True,
                    "model_new_source": kernel.model_new_source,
                    "kernel_name": kernel.name,
                    "cuda_source": preprocessed_cuda,
                    "original_cuda_source": kernel.cuda_source,
                    "kernel_functions": kernel.kernel_functions,
                    "compiled_functions": compiled_functions,
                    "cpp_wrapper": cpp_wrapper_result,
                    "gpu_id": gpu_id,
                    "has_cpp_wrapper": cpp_wrapper_result is not None
                }
                
                logger.info(f"Successfully compiled kernel {kernel.name} with {len(compiled_functions)} functions")
                return result
                
        except Exception as e:
            logger.error(f"Compilation failed for kernel {kernel.name}: {e}")
            return {
                "compilation_successful": False,
                "error": str(e),
                "kernel_name": kernel.name
            }
    
    def _preprocess_cuda_source(self, cuda_source: str) -> str:
        """Preprocess CUDA source for CuPy compatibility"""
        processed = cuda_source
        
        # Remove ALL PyTorch-specific includes
        pytorch_includes = [
            '#include <torch/extension.h>',
            '#include <c10/cuda/CUDAException.h>',
        ]
        
        for include in pytorch_includes:
            processed = processed.replace(include, '')
        
        # Remove PyTorch macros and function calls
        pytorch_patterns = [
            r'#define CHECK_CUDA.*?\n',
            r'#define CHECK_CONTIGUOUS.*?\n', 
            r'#define CHECK_INPUT.*?\n',
            r'inline unsigned int cdiv.*?\n',
            r'TORCH_CHECK\([^)]*\);',
            r'CHECK_INPUT\([^)]*\);',
            r'C10_CUDA_KERNEL_LAUNCH_CHECK\(\);',
        ]
        
        for pattern in pytorch_patterns:
            processed = re.sub(pattern, '', processed, flags=re.DOTALL)
        
        # Clean up whitespace
        processed = re.sub(r'\n\s*\n\s*\n', '\n\n', processed)
        processed = processed.strip()
        
        logger.debug(f"Final preprocessed CUDA source ({len(processed)} chars): {processed[:300]}...")
        return processed
    
    def _compile_cpp_wrapper(self, kernel: ExtractedKernel) -> Optional[Dict[str, Any]]:
        """
        Transform and compile C++ wrapper using load_inline
        
        Args:
            kernel: ExtractedKernel with C++ wrapper information
            
        Returns:
            Dictionary with compiled transformed C++ wrapper function
        """
        try:
            if not kernel.cpp_wrapper_source or not kernel.cpp_function_name:
                logger.error("No C++ wrapper source or function name found")
                return None
                
            logger.info(f"Transforming and compiling C++ wrapper for kernel: {kernel.name}")
            
            # Step 1: Transform C++ wrapper using regex patterns
            transformed_cpp_source = self.cpp_wrapper.transform_wrapper(
                kernel.cpp_wrapper_source, 
                kernel.cpp_function_name
            )
            
            if not transformed_cpp_source:
                logger.error(f"Failed to transform C++ wrapper for {kernel.name}")
                return None
            
            logger.debug(f"Transformed C++ source:\n{transformed_cpp_source}")
            
            # Step 2: Compile transformed wrapper with load_inline
            # Ensure build directory exists
            build_dir = os.path.join(self.cache_dir, kernel.hash)
            os.makedirs(build_dir, exist_ok=True)
            
            cpp_extension = load_inline(
                name=f"{kernel.name}_transformed_wrapper",
                cpp_sources=transformed_cpp_source,
                functions=[kernel.cpp_function_name],  # Use original function name
                verbose=False,  # Reduce verbosity for cleaner logs
                extra_cflags=["-O2"],
                extra_ldflags=["-lcuda"],  # Link CUDA driver API
                with_cuda=True,  # Enable CUDA support for driver API
                build_directory=build_dir
            )
            
            result = {
                "cpp_extension": cpp_extension,
                "function_name": kernel.cpp_function_name,
                "transformed_source": transformed_cpp_source,
                "compilation_successful": True
            }
            
            # Add ModelNew source for execution
            result["model_new_source"] = kernel.model_new_source
            
            logger.info(f"Successfully compiled C++ wrapper for {kernel.name}")
            logger.debug(f"############### Extracted CUDA #####################\n{kernel.cuda_source}\n")
            logger.debug(f"############### Extracted C++ Wrapper #####################\n{transformed_cpp_source}\n")
            logger.debug(f"############### Extracted ModelNew #####################\n{kernel.model_new_source}\n")
            
            return result
        except Exception as e:
            logger.error(f"C++ wrapper transformation/compilation failed for {kernel.name}: {e}")
            import traceback
            logger.error(f"Full stack trace: {traceback.format_exc()}")
            return None
