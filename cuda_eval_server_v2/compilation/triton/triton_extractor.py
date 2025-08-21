"""
Triton kernel extractor for parsing and extracting Triton JIT kernels from source code
"""

import ast
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TritonKernelInfo:
    """Information about an extracted Triton kernel"""
    name: str
    jit_function_src: str  # The @triton.jit decorated function
    forward_function: Optional[Dict[str, Any]] = None  # The forward/wrapper function
    input_args: List[str] = field(default_factory=list)  # JIT function input arguments
    constexpr_params: Dict[str, Any] = field(default_factory=dict)  # Constexpr params with values
    grid_config: Optional[Dict[str, Any]] = None  # Grid configuration details
    invocation_args: Optional[List[str]] = None  # Arguments at kernel invocation
    traced_values: Optional[Dict[str, Any]] = None  # Traced parameter values
    metadata: Dict[str, Any] = field(default_factory=dict)


class TritonKernelExtractor:
    """Extract Triton kernels from source code with enhanced parameter tracing"""
    
    def __init__(self):
        self.logger = logger
        
    def extract(self, source_code: str) -> Optional[TritonKernelInfo]:
        """
        Extract Triton kernel information from source code
        
        Args:
            source_code: Python source code containing Triton kernel
            
        Returns:
            TritonKernelInfo with extracted kernel details or None if not found
        """
        try:
            # Parse the source code
            tree = ast.parse(source_code)
            
            # Find @triton.jit decorated functions (assume only one per file)
            jit_functions = self._find_triton_jit_functions(tree)
            
            if not jit_functions:
                self.logger.warning("No @triton.jit decorated functions found")
                return None
            
            if len(jit_functions) > 1:
                self.logger.warning(f"Multiple @triton.jit functions found, using first: {jit_functions[0][0]}")
            
            # Use the first JIT function found
            jit_func_name, jit_func_node = jit_functions[0]
            
            # Extract JIT function source
            jit_func_source = self._extract_function_source(source_code, jit_func_node)
            
            # Extract input arguments from JIT function
            input_args = self._extract_input_args(jit_func_node)
            
            # Find kernel invocation with grid
            invocation_info = self._find_kernel_invocation(tree, source_code, jit_func_name)
            
            # Extract constexpr parameters with attempted value resolution
            constexpr_params = self._extract_constexpr_params_with_values(
                jit_func_node, tree, invocation_info
            )
            
            # Trace parameter values if invocation was found
            traced_values = {}
            if invocation_info and invocation_info.get('wrapper_node'):
                traced_values = self._trace_parameter_values(
                    tree, 
                    invocation_info['wrapper_node'],
                    invocation_info.get('invocation_args', []),
                    source_code
                )
            
            # Extract forward/wrapper function
            forward_func_info = None
            if invocation_info and invocation_info.get('wrapper_node'):
                forward_func_info = {
                    'func_name': invocation_info.get('wrapper_name'),
                    'has_input_args': invocation_info.get('wrapper_node_has_input_args')
                }

            return TritonKernelInfo(
                name=jit_func_name,
                jit_function_src=jit_func_source,
                forward_function=forward_func_info,
                input_args=input_args,
                constexpr_params=constexpr_params,
                grid_config=invocation_info.get('grid_config') if invocation_info else None,
                invocation_args=invocation_info.get('invocation_args') if invocation_info else None,
                traced_values=traced_values,
                metadata={
                    'jit_func_name': jit_func_name,
                    'has_forward': forward_func_info is not None,
                    'has_invocation': invocation_info is not None,
                    'num_constexpr': len(constexpr_params),
                    'num_traced_values': len(traced_values)
                }
            )
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error parsing Triton code: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting Triton kernel: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _find_triton_jit_functions(self, tree: ast.AST) -> List[Tuple[str, ast.FunctionDef]]:
        """Find all @triton.jit decorated functions"""
        jit_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check decorators
                for decorator in node.decorator_list:
                    if self._is_triton_jit_decorator(decorator):
                        jit_functions.append((node.name, node))
                        break
        
        return jit_functions
    
    def _is_triton_jit_decorator(self, decorator: ast.AST) -> bool:
        """Check if a decorator is @triton.jit or similar"""
        # Handle different decorator patterns
        if isinstance(decorator, ast.Name):
            # Simple decorator like @jit
            return decorator.id == 'jit'
        elif isinstance(decorator, ast.Attribute):
            # Attribute decorator like @triton.jit
            if isinstance(decorator.value, ast.Name):
                return (decorator.value.id == 'triton' and 
                       decorator.attr == 'jit')
        elif isinstance(decorator, ast.Call):
            # Decorator with arguments like @triton.jit(...)
            if isinstance(decorator.func, ast.Attribute):
                if isinstance(decorator.func.value, ast.Name):
                    return (decorator.func.value.id == 'triton' and 
                           decorator.func.attr == 'jit')
            elif isinstance(decorator.func, ast.Name):
                return decorator.func.id == 'jit'
        
        return False
    
    def _extract_input_args(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract all input argument names from JIT function"""
        return [arg.arg for arg in func_node.args.args]
    
    def _find_kernel_invocation(self, tree: ast.AST, source_code: str, 
                                kernel_name: str) -> Optional[Dict[str, Any]]:
        """
        Find kernel invocation with grid configuration
        
        Returns dict with:
        - wrapper_node: The function containing the invocation
        - wrapper_name: Name of the wrapper function
        - grid_config: Grid configuration details
        - invocation_args: Arguments passed to kernel
        - line_number: Line where invocation occurs
        """
        # First try AST-based search
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip the JIT function itself
                if node.name == kernel_name:
                    continue
                    
                # Search for kernel invocation in this function
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Subscript):
                        # Check if this is kernel[grid](...) pattern
                        if isinstance(subnode.value, ast.Name) and subnode.value.id == kernel_name:
                            # Found kernel[...] pattern
                            grid_expr = self._extract_grid_expression(subnode.slice)
                            
                            # Find the Call node that uses this subscript
                            parent_call = self._find_parent_call(node, subnode)
                            invocation_args = []
                            if parent_call:
                                invocation_args = self._extract_call_args(parent_call)
                            
                            return {
                                'wrapper_node': node,
                                'wrapper_name': node.name,
                                'wrapper_node_has_input_args': len(node.args.args) > 1 if len(node.args.args) > 0 and node.args.args[0].arg == 'self' else False,
                                'grid_config': self._parse_grid_config(grid_expr),
                                'invocation_args': invocation_args,
                                'line_number': subnode.lineno if hasattr(subnode, 'lineno') else None
                            }
        
        # Fallback to regex-based search for more complex patterns
        return self._find_kernel_invocation_regex(source_code, kernel_name)
    
    def _find_kernel_invocation_regex(self, source_code: str, 
                                      kernel_name: str) -> Optional[Dict[str, Any]]:
        """Regex-based fallback for finding kernel invocation"""
        # Pattern to match kernel_name[grid](...) 
        pattern = rf'{kernel_name}\s*\[(.*?)\]\s*\((.*?)\)'
        
        for match in re.finditer(pattern, source_code, re.DOTALL):
            grid_expr = match.group(1).strip()
            args_expr = match.group(2).strip()
            
            # Try to find which function this is in
            lines_before = source_code[:match.start()].split('\n')
            
            # Look backward for function definition
            wrapper_name = None
            for i in range(len(lines_before) - 1, -1, -1):
                if 'def ' in lines_before[i]:
                    func_match = re.match(r'\s*def\s+(\w+)', lines_before[i])
                    if func_match:
                        wrapper_name = func_match.group(1)
                        break
            
            # Parse arguments
            invocation_args = [arg.strip() for arg in args_expr.split(',') if arg.strip()]
            
            return {
                'wrapper_node': None,  # Can't get AST node from regex
                'wrapper_name': wrapper_name,
                'grid_config': self._parse_grid_config(grid_expr),
                'invocation_args': invocation_args,
                'line_number': len(lines_before) + 1
            }
        
        return None
    
    def _extract_grid_expression(self, slice_node: ast.AST) -> str:
        """Extract grid expression from subscript slice"""
        if hasattr(ast, 'unparse'):
            return ast.unparse(slice_node)
        
        # Fallback for older Python versions
        if isinstance(slice_node, ast.Tuple):
            return f"({', '.join(self._ast_to_str(e) for e in slice_node.elts)})"
        else:
            return self._ast_to_str(slice_node)
    
    def _ast_to_str(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Num):  # Python < 3.8
            return str(node.n)
        elif isinstance(node, ast.Lambda):
            return f"lambda {', '.join(a.arg for a in node.args.args)}: ..."
        elif isinstance(node, ast.Call):
            func_name = self._ast_to_str(node.func)
            args = ', '.join(self._ast_to_str(arg) for arg in node.args)
            return f"{func_name}({args})"
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_str(node.left)
            right = self._ast_to_str(node.right)
            op = self._get_binop_str(node.op)
            return f"{left} {op} {right}"
        else:
            return "..."
    
    def _get_binop_str(self, op: ast.AST) -> str:
        """Get string representation of binary operator"""
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
        }
        return op_map.get(type(op), '?')
    
    def _parse_grid_config(self, grid_expr: str) -> Dict[str, Any]:
        """Parse grid configuration from expression"""
        config = {
            'expression': grid_expr,
            'type': 'unknown',
            'dimensions': None
        }
        
        # Detect lambda
        if 'lambda' in grid_expr:
            config['type'] = 'lambda'
            # Try to extract META parameter usage
            if 'META' in grid_expr:
                config['uses_meta'] = True
        # Detect tuple
        elif grid_expr.startswith('(') or ',' in grid_expr:
            config['type'] = 'tuple'
            # Count dimensions
            config['dimensions'] = grid_expr.count(',') + 1
        # Single value
        else:
            config['type'] = 'value'
            config['dimensions'] = 1
        
        return config
    
    def _find_parent_call(self, func_node: ast.FunctionDef, 
                         subscript_node: ast.Subscript) -> Optional[ast.Call]:
        """Find the Call node that contains the subscript"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Subscript) and node.func == subscript_node:
                    return node
        return None
    
    def _extract_call_args(self, call_node: ast.Call) -> List[str]:
        """Extract argument names/expressions from a Call node"""
        args = []
        for arg in call_node.args:
            if hasattr(ast, 'unparse'):
                args.append(ast.unparse(arg))
            else:
                args.append(self._ast_to_str(arg))
        return args
    
    def _extract_constexpr_params_with_values(self, func_node: ast.FunctionDef,
                                             tree: ast.AST,
                                             invocation_info: Optional[Dict]) -> Dict[str, Any]:
        """Extract constexpr parameters and attempt to resolve their values"""
        constexpr_params = {}
        
        # First identify constexpr parameters
        for arg in func_node.args.args:
            is_constexpr = False
            
            # Check annotation
            if arg.annotation:
                annotation_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                if 'constexpr' in annotation_str.lower():
                    is_constexpr = True
            
            # Also check common constexpr naming patterns
            if arg.arg.isupper() or arg.arg.endswith('_SIZE') or arg.arg.startswith('BLOCK_'):
                is_constexpr = True
            
            if is_constexpr:
                # Try to find the value
                value = self._resolve_constexpr_value(arg.arg, tree, invocation_info)
                constexpr_params[arg.arg] = value
        
        return constexpr_params
    
    def _resolve_constexpr_value(self, param_name: str, tree: ast.AST,
                                 invocation_info: Optional[Dict]) -> Any:
        """Try to resolve the value of a constexpr parameter"""
        # Look for direct assignments in module scope
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == param_name:
                        return self._eval_ast_value(node.value)
        
        # If we have invocation info, try to trace from there
        if invocation_info and invocation_info.get('invocation_args'):
            # This would require more complex tracing
            pass
        
        return None  # Could not resolve
    
    def _trace_parameter_values(self, tree: ast.AST, wrapper_node: ast.FunctionDef,
                               invocation_args: List[str], 
                               source_code: str) -> Dict[str, Any]:
        """
        Trace parameter values from invocation back to their definitions
        """
        traced = {}
        
        # Build a map of all assignments in the wrapper function
        assignments = {}
        for node in ast.walk(wrapper_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
        
        # Trace each invocation argument
        for arg_expr in invocation_args:
            # Parse the argument expression
            if '=' in arg_expr:  # Skip keyword arguments for now
                continue
                
            # Simple variable reference
            if arg_expr in assignments:
                value = self._eval_ast_value(assignments[arg_expr])
                if value is not None:
                    traced[arg_expr] = value
            
            # Shape access like x.shape[0]
            elif '.shape' in arg_expr:
                base_var = arg_expr.split('.')[0]
                traced[arg_expr] = f"shape_of_{base_var}"
            
            # Direct literals
            elif arg_expr.isdigit():
                traced[arg_expr] = int(arg_expr)
            elif arg_expr in ('True', 'False'):
                traced[arg_expr] = arg_expr == 'True'
        
        return traced
    
    def _eval_ast_value(self, node: ast.AST) -> Any:
        """Safely evaluate an AST node to get its value"""
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            return node.value if hasattr(node, 'value') else None
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner_val = self._eval_ast_value(node.operand)
            return -inner_val if inner_val is not None else None
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_value(node.left)
            right = self._eval_ast_value(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
        elif isinstance(node, ast.Call):
            # Handle simple function calls like cdiv(a, b)
            if hasattr(node.func, 'id'):
                func_name = node.func.id
                if func_name == 'cdiv' and len(node.args) == 2:
                    a = self._eval_ast_value(node.args[0])
                    b = self._eval_ast_value(node.args[1])
                    if a is not None and b is not None:
                        return (a + b - 1) // b
        
        return None
    
    def _extract_function_source(self, source_code: str, func_node: ast.FunctionDef) -> str:
        """Extract the source code of a function"""
        # Get line numbers
        start_line = func_node.lineno - 1  # Convert to 0-based
        
        # Find the end of the function
        end_line = start_line
        for node in ast.walk(func_node):
            if hasattr(node, 'lineno'):
                end_line = max(end_line, node.lineno - 1)
        
        # Extract lines
        lines = source_code.split('\n')
        
        # Include decorators
        decorator_start = start_line
        if func_node.decorator_list:
            decorator_start = min(d.lineno - 1 for d in func_node.decorator_list 
                                 if hasattr(d, 'lineno'))
        
        # Extract function with decorators
        func_lines = lines[decorator_start:end_line + 1]
        
        # Find the actual end of the function (handle multi-line definitions)
        func_source = '\n'.join(func_lines)
        
        # Extend if we didn't capture the whole function
        if end_line + 1 < len(lines):
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            for i in range(end_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                    break
                func_lines.append(line)
            func_source = '\n'.join(func_lines)
        
        return func_source
    
    def _find_forward_function(self, tree: ast.AST, source_code: str) -> Optional[str]:
        """Find the forward function in ModelNew class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for ModelNew or Model class
                if 'Model' in node.name:
                    # Find forward method
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                            return self._extract_function_source(source_code, item)
        
        # Try to find standalone forward function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'forward':
                # Check if it's not inside a class
                parent = self._get_parent_node(tree, node)
                if not isinstance(parent, ast.ClassDef):
                    return self._extract_function_source(source_code, node)
        
        return None
    
    def _get_parent_node(self, tree: ast.AST, target_node: ast.AST) -> Optional[ast.AST]:
        """Get the parent node of a target node in the AST"""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child == target_node:
                    return node
        return None