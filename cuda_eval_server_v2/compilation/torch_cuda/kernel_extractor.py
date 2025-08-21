"""
CUDA kernel extractor for TORCH_CUDA code
Extracts embedded CUDA kernels from PyTorch models
"""

import re
import hashlib
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedKernel:
    """Container for extracted CUDA kernel information"""
    name: str
    cuda_source: str
    cpp_wrapper_source: str
    cpp_function_name: str
    kernel_functions: List[str]
    model_new_source: str
    hash: str
    metadata: Dict[str, Any] = None


class TorchCudaExtractor:
    """
    Extracts CUDA kernels from PyTorch models with embedded CUDA code
    Based on SimpleCudaExtractor from simple_cupy_compiler.py
    """
    
    def extract(self, source_code: str) -> Optional[ExtractedKernel]:
        """
        Extract CUDA kernel from TORCH_CUDA source code
        
        Args:
            source_code: Python source with embedded CUDA
            
        Returns:
            ExtractedKernel with CUDA source and metadata, or None if extraction fails
        """
        try:
            logger.info(f"############### Custom Code #####################\n{source_code}\n")
            logger.info(f"############### Custom Code End ##################")
            # Find load_inline calls
            load_inline_pattern = r'load_inline\s*\((.*?)\)'
            matches = list(re.finditer(load_inline_pattern, source_code, re.DOTALL))
            
            if not matches:
                logger.warning("No load_inline call found in source_code")
                return None
            
            # Parse the first load_inline call
            args_str = matches[0].group(1)
            
            # Extract name from load_inline call
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', args_str)
            name = name_match.group(1) if name_match else "cuda_kernel"
            
            # Extract cuda_sources variable name
            cuda_sources_match = re.search(r'cuda_sources\s*=\s*([^,\)]+)', args_str)
            if not cuda_sources_match:
                logger.error("No cuda_sources found in load_inline call")
                return None
            
            cuda_var = cuda_sources_match.group(1).strip()
            
            # Extract cpp_sources variable name
            cpp_sources_match = re.search(r'cpp_sources\s*=\s*([^,\)]+)', args_str)
            if not cpp_sources_match:
                logger.error("No cpp_sources found in load_inline call")
                return None
            
            cpp_var = cpp_sources_match.group(1).strip()
            
            # Extract functions list (C++ wrapper functions)
            functions_match = re.search(r'functions\s*=\s*\[(.*?)\]', args_str)
            cpp_functions = []
            if functions_match:
                functions_str = functions_match.group(1)
                cpp_functions = [f.strip().strip('\'"') for f in functions_str.split(',') if f.strip()]
            
            if not cpp_functions:
                logger.error("No C++ functions found in load_inline call")
                return None
            
            # Extract CUDA source content
            cuda_source = self._extract_cuda_source(cuda_var, source_code)
            if not cuda_source:
                logger.error(f"Failed to extract CUDA source from variable: {cuda_var}")
                return None
            
            # Extract C++ wrapper source content
            cpp_wrapper_source = self._extract_cpp_wrapper_source(cpp_var, source_code)
            if not cpp_wrapper_source:
                logger.error(f"Failed to extract C++ wrapper source from variable: {cpp_var}")
                return None
            
            # Find __global__ kernel function names in the CUDA source
            global_functions = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', cuda_source)
            if not global_functions:
                logger.error("No __global__ functions found in CUDA source")
                logger.debug(f"CUDA source preview: {cuda_source}\n")
                return None
            
            logger.info(f"Kernel name: {name}\n CUDA source variable: {cuda_var}\n C++ source variable: {cpp_var}\n C++ functions: {cpp_functions}\n __global__ kernel functions: {global_functions}\n")
            
            # Extract C++ function signature from the wrapper
            cpp_function_name = cpp_functions[0]  # Use first function
            cpp_signature = self._extract_cpp_function_signature(cpp_wrapper_source, cpp_function_name)

            # Extract torch module in generated code
            model_new_source = self._extract_model_new(source_code)
            if not model_new_source:
                logger.error("No ModelNew torch module found in generated source")
                return None
            
            # Create hash for caching (include both CUDA and C++ sources)
            combined_source = cuda_source + cpp_wrapper_source + model_new_source
            kernel_hash = hashlib.sha256(combined_source.encode()).hexdigest()[:16]
            
            return ExtractedKernel(
                name=name,
                cuda_source=cuda_source,
                cpp_wrapper_source=cpp_wrapper_source,
                cpp_function_name=cpp_function_name,
                kernel_functions=global_functions,
                model_new_source=model_new_source,
                hash=kernel_hash,
                metadata={
                    "source_length": len(cuda_source),
                    "num_kernels": len(global_functions)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to extract CUDA kernel: {e}")
            return None
    
    def _extract_cuda_source(self, var_name: str, code: str) -> Optional[str]:
        """
        Extracts the CUDA kernel (with its #includes) from the first
        triple-quoted block in `source`.
        """
        # 1) Find the first triple‑quoted block:
        block_pattern = re.compile(r'([\'"]{3})(.*?)(\1)', re.DOTALL)
        m = block_pattern.search(code)
        if not m:
            raise ValueError("No triple-quoted block found")
        block = m.group(2)

        # 2) Locate the start of includes and the kernel keyword
        inc_pos = block.find('#include')
        kernel_pos = block.find('__global__', inc_pos)
        if inc_pos < 0 or kernel_pos < 0:
            raise ValueError("CUDA includes or kernel not found")

        # 3) Scan from kernel_pos to match braces for the kernel function
        snippet = block[inc_pos:]
        brace = 0
        end = None
        for idx, ch in enumerate(snippet[kernel_pos-inc_pos:], start=kernel_pos-inc_pos):
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    end = idx
                    break
        if end is None:
            raise ValueError("Kernel closing brace not found")

        # 4) Combine includes + kernel body
        includes = snippet[:kernel_pos-inc_pos]
        kernel = snippet[kernel_pos-inc_pos:end+1]
        return (includes + 'extern "C" ' + kernel).strip('\n')


    def _extract_cpp_wrapper_source(self, var_name: str, code: str) -> Optional[str]:
        """
        Extract C++ wrapper source from variable assignment
        Looks for patterns like: cpp_src = "torch::Tensor func_name(...);"
        If only a signature is found, looks for the full implementation in the triple-quoted source.
        """
        # Look for variable assignment patterns
        patterns = [
            # Pattern 1: var_name = "single line"
            rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']',
            # Pattern 2: var_name = ("multi line")
            rf'{re.escape(var_name)}\s*=\s*\(\s*["\']([^"\']+)["\']\s*\)',
            # Pattern 3: var_name = """triple quoted"""
            rf'{re.escape(var_name)}\s*=\s*["\']{{3}}(.*?)["\']{{3}}',
        ]
        
        cpp_source = None
        for pattern in patterns:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                cpp_source = match.group(1).strip()
                break
        
        if not cpp_source:
            logger.error(f"Could not find C++ source variable {var_name}")
            return None
        
        # Check if we only got a function signature (ends with ';' and relatively short)
        # If so, look for the full implementation in the triple-quoted source block
        if cpp_source.endswith(';') and len(cpp_source) < 200 and '{' not in cpp_source:
            # Extract the function name from the signature
            func_name_match = re.search(r'(\w+)\s*\([^)]*\)\s*;?\s*$', cpp_source)
            if func_name_match:
                func_name = func_name_match.group(1)
                
                # Look for the full function implementation in the triple-quoted source block
                full_impl = self._extract_cpp_function_from_source_block(code, func_name)
                if full_impl:
                    return full_impl
                else:
                    logger.warning(f"Could not find full implementation for {func_name}, using signature")
        
        return cpp_source
            
    def _extract_cpp_function_from_source_block(self, code: str, func_name: str) -> Optional[str]:
        """
        Extract the full C++ function implementation from the triple-quoted source block
        Including necessary headers for compilation
        """
        # Find the first triple-quoted block (which should contain both CUDA and C++ code)
        block_pattern = re.compile(r'([\'"]{3})(.*?)(\1)', re.DOTALL)
        match = block_pattern.search(code)
        if not match:
            logger.warning("No triple-quoted block found for C++ function extraction")
            return None
        
        source_block = match.group(2)
        
        # Extract headers from the source block
        headers = []
        header_pattern = r'^#include\s*[<"][^>"]*[>"]'
        for line in source_block.split('\n'):
            line = line.strip()
            if re.match(header_pattern, line):
                headers.append(line)
        
        # Look for the C++ function definition pattern: return_type func_name(...) {
        # We need to be more flexible with return types and parameter lists
        func_pattern = rf'(\w+::\w+\s+{re.escape(func_name)}\s*\([^{{]*\)\s*\{{.*?\n\}})'
        func_match = re.search(func_pattern, source_block, re.DOTALL | re.MULTILINE)
        
        if func_match:
            full_function = func_match.group(1).strip()
            
            # Combine headers with function implementation
            if headers:
                combined_source = '\n'.join(headers) + '\n\n' + full_function
            else:
                combined_source = full_function
            
            return combined_source
        else:
            logger.warning(f"Could not find full function implementation for {func_name} in source block")
            return None
                
    def _extract_cpp_function_signature(self, cpp_source: str, function_name: str) -> str:
        """
        Extract C++ function signature from the wrapper source
        For example: "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight);"
        """
        # Look for function declaration/definition patterns
        patterns = [
            # Pattern 1: Full function signature
            rf'(torch::Tensor\s+{re.escape(function_name)}\s*\([^;{{]+\))\s*[;{{]',
            # Pattern 2: Just the signature line
            rf'({re.escape(function_name)}\s*\([^;{{]+\))\s*[;{{]',
            # Pattern 3: Any return type
            rf'(\w+::\w+\s+{re.escape(function_name)}\s*\([^;{{]+\))\s*[;{{]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cpp_source, re.MULTILINE | re.DOTALL)
            if match:
                signature = match.group(1).strip()
                # Add semicolon if not present
                if not signature.endswith(';'):
                    signature += ';'
                return signature

        # Final fallback: construct a basic signature
        signature = f"torch::Tensor {function_name}(torch::Tensor input, torch::Tensor weight);"
        logger.warning(f"Could not extract signature, using fallback: {signature}")
        return signature
    
    def _extract_model_new(self, code: str) -> str:
        """
        Extract ModelNew class and imports from source code string.
        Filters out CUDA kernel source, compilation code, and intermediate variables.
        
        Args:
            code: Python source code string containing ModelNew class
            
        Returns:
            Filtered source code with only imports and ModelNew class
        """
        lines = code.split('\n')
        
        # Step 1: Extract import statements from the beginning
        imports = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(line)
            elif stripped and not stripped.startswith('#'):
                # Stop at first non-import, non-comment line
                break
        
        # Step 2: Find ModelNew class definition
        class_start = None
        for i, line in enumerate(lines):
            if re.match(r'^\s*class\s+ModelNew\s*\(', line):
                class_start = i
                break
        
        if class_start is None:
            logger.warning("ModelNew class definition not found")
            return ""
        
        # Step 3: Extract complete class definition by tracking indentation
        class_lines = [lines[class_start]]
        base_indent = len(lines[class_start]) - len(lines[class_start].lstrip())
        
        for i in range(class_start + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                # Include empty lines within the class
                class_lines.append(line)
            else:
                line_indent = len(line) - len(line.lstrip())
                if line_indent > base_indent:
                    # This line is part of the class (indented more than class definition)
                    class_lines.append(line)
                else:
                    # This line is at or before class level, class definition ends
                    break
        
        # Step 4: Combine imports and class with proper spacing
        result_parts = []
        if imports:
            result_parts.extend(imports)
            result_parts.append('')  # Empty line after imports
        
        result_parts.extend(class_lines)
        
        return '\n'.join(result_parts)
