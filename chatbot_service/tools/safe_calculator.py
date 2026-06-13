"""
Safe Calculator - AST-based mathematical expression evaluator.

Replaces eval() with safe AST parsing to prevent code injection attacks.

Security features:
- Only allows numeric constants and basic operators
- No function calls except whitelisted math functions
- Overflow protection (MAX_RESULT limit)
- Recursion depth limit to prevent stack overflow
- Execution timeout (thread-based) to prevent CPU exhaustion
- No access to builtins or other modules
"""

import ast
import operator
import math
import sys
import threading
from typing import Union
from contextlib import contextmanager

class SafeCalculator:
    """
    Safe mathematical expression evaluator using AST parsing.
    
    Supported operations:
    - Basic: +, -, *, /, //, %, **
    - Unary: +x, -x
    - Functions: abs, round, min, max, sum, sqrt, pow, sin, cos, tan
    - Constants: pi, e
    
    Example:
        calc = SafeCalculator()
        calc.evaluate("2 + 2")  # 4
        calc.evaluate("(10 - 3) * 4")  # 28
        calc.evaluate("sqrt(16)")  # 4.0
        calc.evaluate("round(3.7)")  # 4
    """
    
    # Binary operators supported
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }
    
    # Unary operators supported
    UNARY_OPERATORS = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Safe functions allowed
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'pow': pow,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'ceil': math.ceil,
        'floor': math.floor,
    }
    
    # Safe constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }
    
    # Limits for safety
    MAX_RESULT = 1e18  # Prevent overflow attacks
    MAX_POWER = 100    # Prevent 10**1000 type attacks
    MAX_RECURSION_DEPTH = 100  # Prevent stack overflow from deeply nested expressions
    MAX_NODES = 1000  # Prevent AST with too many operations
    EVAL_TIMEOUT = 5  # Seconds before evaluation is aborted
    
    def __init__(self):
        """Initialize calculator with recursion tracking."""
        self._recursion_depth = 0
        self._node_count = 0
    
    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Math expression string (e.g., "2 + 2", "sqrt(16)")
            
        Returns:
            Numeric result (int or float)
            
        Raises:
            ValueError: If expression is invalid, unsafe, or hits complexity limits
            TimeoutError: If evaluation exceeds EVAL_TIMEOUT seconds
        """
        if not expression or not expression.strip():
            raise ValueError("Empty expression")
        
        # Limit expression length
        if len(expression) > 500:
            raise ValueError("Expression too long (max 500 characters)")
        
        # Reset recursion tracking
        self._recursion_depth = 0
        self._node_count = 0
        
        # Strip whitespace to prevent ast.parse indentation errors
        expression = expression.strip()
        
        try:
            tree = ast.parse(expression, mode='eval')
            
            # Check total node count to prevent huge AST attacks
            if self._count_nodes(tree) > self.MAX_NODES:
                raise ValueError(f"Expression too complex (max {self.MAX_NODES} operations)")
            
            # Evaluate with a thread-based timeout (cross-platform)
            result_holder: list = []
            error_holder: list = []

            def _target():
                try:
                    result_holder.append(self._eval_node(tree.body))
                except Exception as exc:
                    error_holder.append(exc)

            worker = threading.Thread(target=_target, daemon=True)
            worker.start()
            worker.join(timeout=self.EVAL_TIMEOUT)

            if worker.is_alive():
                # Thread is still running — evaluation timed out
                raise TimeoutError(
                    f"Expression evaluation timed out after "
                    f"{self.EVAL_TIMEOUT}s"
                )

            if error_holder:
                raise error_holder[0]

            return result_holder[0]
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {e}")
        except (TypeError, KeyError) as e:
            raise ValueError(f"Invalid expression: {e}")
        except RecursionError:
            raise ValueError("Expression too deeply nested (recursion limit exceeded)")
    
    def _count_nodes(self, node) -> int:
        """Count total nodes in AST to prevent complexity attacks."""
        count = 1
        for child in ast.walk(node):
            count += 1
        return count
    
    def _eval_node(self, node) -> Union[int, float]:
        """
        Recursively evaluate an AST node.
        
        Includes recursion depth tracking to prevent stack exhaustion.
        """
        # Track and enforce recursion depth
        self._recursion_depth += 1
        if self._recursion_depth > self.MAX_RECURSION_DEPTH:
            self._recursion_depth -= 1
            raise ValueError(f"Expression too deeply nested (max depth {self.MAX_RECURSION_DEPTH})")
        
        try:
            result = self._eval_node_impl(node)
            self._recursion_depth -= 1
            return result
        except Exception as e:
            self._recursion_depth -= 1
            raise
    
    def _eval_node_impl(self, node) -> Union[int, float]:
        """Implementation of node evaluation (separated for cleaner recursion handling)."""
        # Numeric constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Invalid constant type: {type(node.value).__name__}")
        
        # For Python 3.7 compatibility (Num is deprecated but may exist)
        if isinstance(node, ast.Num):
            return node.n
        
        # Binary operations: +, -, *, /, etc.
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            
            # Special handling for power to prevent huge numbers
            if isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > self.MAX_POWER:
                    raise ValueError(f"Power exponent too large (max {self.MAX_POWER})")
            
            # Prevent division by zero
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("Division by zero")
            
            result = op_func(left, right)
            
            # Check result bounds
            if isinstance(result, (int, float)) and abs(result) > self.MAX_RESULT:
                raise ValueError(f"Result too large (exceeds {self.MAX_RESULT:.0e})")
            
            return result
        
        # Unary operations: -x, +x
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            
            op_func = self.UNARY_OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            
            return op_func(operand)
        
        # Function calls: sqrt(16), round(3.7), etc.
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                if func_name in self.FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    
                    # Validate arguments
                    for arg in args:
                        if isinstance(arg, (int, float)) and abs(arg) > self.MAX_RESULT:
                            raise ValueError("Function argument too large")
                    
                    try:
                        result = self.FUNCTIONS[func_name](*args)
                        if isinstance(result, (int, float)) and abs(result) > self.MAX_RESULT:
                            raise ValueError("Function result too large")
                        return result
                    except (ValueError, ZeroDivisionError, OverflowError) as e:
                        raise ValueError(f"Math error in {func_name}(): {e}")
                else:
                    raise ValueError(f"Unknown function: {func_name}")
            else:
                raise ValueError("Complex function calls not supported")
        
        # Variable names (for constants like pi, e)
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.CONSTANTS:
                return self.CONSTANTS[name]
            raise ValueError(f"Unknown variable: {name}")
        
        # Comparison operators (not supported for safety)
        if isinstance(node, ast.Compare):
            raise ValueError("Comparison operators not supported")
        
        # Boolean operators (not supported)
        if isinstance(node, ast.BoolOp):
            raise ValueError("Boolean operators not supported")
        
        # Anything else is not allowed
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# Singleton instance
_calculator = SafeCalculator()


def safe_evaluate(expression: str) -> Union[int, float]:
    """
    Convenience function for safe expression evaluation.
    
    Args:
        expression: Math expression to evaluate
        
    Returns:
        Numeric result
        
    Raises:
        ValueError: If expression is invalid
    """
    return _calculator.evaluate(expression)


# For testing
if __name__ == "__main__":
    calc = SafeCalculator()
    
    # Test cases
    tests = [
        ("2 + 2", 4),
        ("10 - 3", 7),
        ("4 * 5", 20),
        ("20 / 4", 5.0),
        ("2 ** 10", 1024),
        ("17 % 5", 2),
        ("17 // 5", 3),
        ("(2 + 3) * 4", 20),
        ("-5", -5),
        ("+3", 3),
        ("abs(-5)", 5),
        ("round(3.7)", 4),
        ("min(1, 5, 3)", 1),
        ("max(1, 5, 3)", 5),
        ("sqrt(16)", 4.0),
        ("sin(0)", 0.0),
        ("pi", 3.141592653589793),
    ]
    
    print("SafeCalculator Tests:")
    print("-" * 50)
    
    for expr, expected in tests:
        try:
            result = calc.evaluate(expr)
            status = "✅" if abs(result - expected) < 0.0001 else "❌"
            print(f"{status} {expr} = {result} (expected {expected})")
        except Exception as e:
            print(f"❌ {expr} raised {e}")
    
    # Security tests
    print("\nSecurity Tests (should all fail):")
    print("-" * 50)
    
    security_tests = [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "exec('print(1)')",
        "eval('1+1')",
        "10 ** 1000",
        "lambda x: x",
    ]
    
    for expr in security_tests:
        try:
            result = calc.evaluate(expr)
            print(f"❌ SECURITY ISSUE: {expr} returned {result}")
        except ValueError as e:
            print(f"✅ Blocked: {expr[:30]}... ({e})")
