"""
Regex-based C++ Wrapper Transformer
Transforms PyTorch inline CUDA wrapper functions to use CUDA Driver API with CuPy
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TorchCudaCppWrapperTransformer:
    """Regex-based C++ wrapper transformer"""
    
    def __init__(self):
        self.available = True  # Always available since no external dependencies
    
    def transform_wrapper(self, cpp_source: str, function_name: str) -> Optional[str]:
        """
        Transform C++ wrapper function to use CUDA Driver API using regex patterns
        
        Args:
            cpp_source: Original C++ wrapper source
            function_name: Name of function to transform
            
        Returns:
            Transformed C++ source or None if transformation failed
        """
        try:    
            # Step 1: Update headers
            transformed = self._update_headers(cpp_source)
            
            # Step 2: Transform function signature to add func_ptr parameter
            transformed = self._transform_function_signature(transformed, function_name)
            
            # Step 3: Extract tensor data pointers
            transformed = self._extract_data_pointers(transformed)
            
            # Step 4: Transform kernel launches
            transformed = self._transform_kernel_launches(transformed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Regex transformation failed for {function_name}: {e}")
            return None
    
    def _update_headers(self, source: str) -> str:
        """Update headers to include CUDA Driver API"""
        # Add required headers if not present
        headers_to_add = [
            "#include <cuda.h>",
            "#include <ATen/cuda/CUDAContext.h>"
        ]
        
        # Find the last #include line
        include_pattern = r'#include\s*[<"][^>"]+[>"]'
        includes = list(re.finditer(include_pattern, source))
        
        if includes:
            # Insert after the last include
            last_include = includes[-1]
            insert_pos = last_include.end()
            
            # Check which headers we need to add
            headers_needed = []
            for header in headers_to_add:
                if header not in source:
                    headers_needed.append(header)
            
            if headers_needed:
                new_headers = "\n" + "\n".join(headers_needed)
                source = source[:insert_pos] + new_headers + source[insert_pos:]
        
        return source
    
    def _transform_function_signature(self, source: str, function_name: str) -> str:
        """Add func_ptr parameter to function signature"""
        # Pattern to match function signature
        pattern = rf'(\w+::\w+\s+{re.escape(function_name)}\s*\([^)]*)\)(\s*\{{)'
        
        def replace_signature(match):
            params = match.group(1)
            closing = match.group(2)
            
            # Add func_ptr parameter
            if params.strip().endswith('('):
                # No parameters
                new_params = params + "uint64_t func_ptr"
            else:
                # Has parameters
                new_params = params + ", uint64_t func_ptr"
            
            return new_params + ")" + closing
        
        return re.sub(pattern, replace_signature, source)
    
    def _extract_data_pointers(self, source: str) -> str:
        """Extract tensor data pointers into separate variables"""
        # Find all data_ptr calls and collect unique ones
        data_ptr_pattern = r'(\w+)\.data_ptr<([^>]+)>\(\)'
        data_ptr_calls = re.findall(data_ptr_pattern, source)
        
        # Create unique data pointer variables
        data_ptr_vars = {}
        for tensor_name, data_type in data_ptr_calls:
            if tensor_name not in data_ptr_vars:
                data_ptr_vars[tensor_name] = data_type
        
        if not data_ptr_vars:
            return source
        
        # Find where to insert data pointer declarations (after variable declarations)
        # Look for the first kernel launch or return statement
        kernel_pattern = r'\w+\s*<<<[^>]+>>>\s*\('
        return_pattern = r'return\s+'
        
        kernel_match = re.search(kernel_pattern, source)
        return_match = re.search(return_pattern, source)
        
        # Find insertion point (before first kernel launch or return)
        insert_pos = None
        if kernel_match and return_match:
            insert_pos = min(kernel_match.start(), return_match.start())
        elif kernel_match:
            insert_pos = kernel_match.start()
        elif return_match:
            insert_pos = return_match.start()
        
        if insert_pos is None:
            # Fallback: insert before last closing brace
            last_brace = source.rfind('}')
            if last_brace != -1:
                insert_pos = last_brace
        
        if insert_pos is not None:
            # Generate data pointer declarations (BEFORE replacing the calls)
            declarations = []
            declarations.append("\n    // Extract tensor data pointers")
            for tensor_name, data_type in data_ptr_vars.items():
                # Use the original data_ptr call in the declaration
                declarations.append(f"    {data_type}* {tensor_name}_ptr = {tensor_name}.data_ptr<{data_type}>();")
            declarations.append("")
            
            declaration_text = "\n".join(declarations)
            source = source[:insert_pos] + declaration_text + source[insert_pos:]
            
            # Now replace data_ptr calls with pointer variables (after declarations are inserted)
            # Split the source to avoid replacing our declarations
            lines = source.split('\n')
            for i, line in enumerate(lines):
                # Skip lines that are our declarations (contain "_ptr = ")
                if '_ptr = ' in line and '.data_ptr<' in line:
                    continue
                    
                # Replace data_ptr calls in other lines
                for tensor_name, data_type in data_ptr_vars.items():
                    old_call = f"{tensor_name}.data_ptr<{data_type}>()"
                    new_var = f"{tensor_name}_ptr"
                    lines[i] = lines[i].replace(old_call, new_var)
                    
            source = '\n'.join(lines)
        
        return source
    
    def _transform_kernel_launches(self, source: str) -> str:
        """Transform kernel<<<>>> launches to cuLaunchKernel calls"""
        # Pattern to match kernel launch: kernel_name<<<grid, block>>>(args)
        # Use a more sophisticated approach to handle nested parentheses
        kernel_pattern = r'(\w+)\s*<<<\s*(.+?)\s*>>>\s*\(([^)]*)\)\s*;'
        
        def parse_dim3_expression(expr, source_context):
            """
            Parse a grid/block expression and return (x, y, z) components.
            Handles:
            - dim3 variables: grid, block
            - Inline dim3 construction: dim3(x, y, z)
            - Simple numeric expressions: num_blocks, 256
            """
            expr = expr.strip()
            
            # Check for inline dim3 construction: dim3(x, y, z)
            dim3_inline = re.match(r'dim3\s*\(([^)]+)\)', expr)
            if dim3_inline:
                args = [arg.strip() for arg in dim3_inline.group(1).split(',')]
                # Pad with 1s if fewer than 3 dimensions
                while len(args) < 3:
                    args.append('1')
                return args[0], args[1], args[2]
            
            # Check if it's likely a dim3 variable by looking for its declaration
            # Search for: dim3 <expr>(...) or dim3 <expr> = ...
            dim3_var_pattern = rf'dim3\s+{re.escape(expr)}\s*[=(]'
            if re.search(dim3_var_pattern, source_context):
                # It's a dim3 variable, access its components
                return f'{expr}.x', f'{expr}.y', f'{expr}.z'
            
            # Otherwise, treat as a simple numeric expression (use as X dimension)
            return expr, '1', '1'
        
        def replace_kernel_launch(match):
            kernel_name = match.group(1)
            grid_block_expr = match.group(2).strip()
            args_str = match.group(3).strip()
            
            # Split grid and block expressions, handling nested parentheses
            # This is tricky because we need to respect parentheses in dim3(...)
            paren_depth = 0
            split_pos = -1
            
            for i, char in enumerate(grid_block_expr):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    # Found the comma separating grid and block at depth 0
                    split_pos = i
                    break
            
            if split_pos > 0:
                grid_expr = grid_block_expr[:split_pos].strip()
                block_expr = grid_block_expr[split_pos+1:].strip()
            else:
                # Shouldn't happen, but fallback
                grid_expr = grid_block_expr
                block_expr = '1'
            
            # Parse grid and block dimensions
            grid_x, grid_y, grid_z = parse_dim3_expression(grid_expr, source)
            block_x, block_y, block_z = parse_dim3_expression(block_expr, source)
            
            # Parse arguments
            if args_str:
                args = [arg.strip() for arg in args_str.split(',')]
            else:
                args = []
            
            # Build cuLaunchKernel replacement
            lines = []
            lines.append(f"    // Launch {kernel_name} kernel using CUDA Driver API")
            
            # Build args array
            if args:
                arg_refs = []
                for arg in args:
                    # For pointer variables, use as-is; for others, take address
                    if '_ptr' in arg or arg.endswith('*'):
                        arg_refs.append(f"&{arg}")
                    else:
                        arg_refs.append(f"&{arg}")
                
                args_array = "{ " + ", ".join(arg_refs) + " }"
                lines.append(f"    void* args[] = {args_array};")
            else:
                lines.append("    void* args[] = {};")
            
            lines.append("")
            lines.append("    CUfunction func = reinterpret_cast<CUfunction>(func_ptr);")
            lines.append("    CUstream cuStream = at::cuda::getCurrentCUDAStream();")
            lines.append("    CUresult err = cuLaunchKernel(func,")
            lines.append(f"                                  {grid_x}, {grid_y}, {grid_z},")
            lines.append(f"                                  {block_x}, {block_y}, {block_z},")
            lines.append("                                  0, cuStream,")
            lines.append("                                  args, nullptr);")
            lines.append('    TORCH_CHECK(err == CUDA_SUCCESS, "cuLaunchKernel failed: ", err);')
            
            return "\n".join(lines)
        
        return re.sub(kernel_pattern, replace_kernel_launch, source)
