from __future__ import annotations
import os
import ast
import math
import logging
import urllib.request
import urllib.parse
import operator
import re as _re2
import json as _json
from typing import Optional, Any, Dict, Type, Callable, Tuple

try:
    from modules.circuit_breaker import arxiv_cb, wolfram_cb
except ImportError:
    try:
        from .circuit_breaker import arxiv_cb, wolfram_cb
    except ImportError:
        from circuit_breaker import arxiv_cb, wolfram_cb

logger = logging.getLogger("eliteomni.tools")

class ResourceConstrainedEvaluator:
    MAX_NODES = 150
    MAX_DEPTH = 15
    MAX_POWER = 10000
    MAX_DIGITS = 50
    MAX_RESULT_DIGITS = 40
    MAX_RESULT_MAGNITUDE = 1e50

    OPERATORS: Dict[Type[ast.AST], Callable[..., Any]] = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    FUNCTION_SIGNATURES: Dict[str, Tuple[Callable[..., Any], int]] = {
        'sin': (math.sin, 1), 'cos': (math.cos, 1), 'tan': (math.tan, 1),
        'sqrt': (math.sqrt, 1), 'log': (math.log, 1), 'exp': (math.exp, 1),
        'abs': (abs, 1), 'pi': (lambda: math.pi, 0), 'e': (lambda: math.e, 0)
    }

    def __init__(self):
        self.node_count = 0

    def evaluate(self, node: ast.AST, depth: int = 1) -> Any:
        if depth > self.MAX_DEPTH:
            raise ValueError(f"Expression tree depth exceeds safe limits ({self.MAX_DEPTH}).")
        
        self.node_count += 1
        if self.node_count > self.MAX_NODES:
            raise ValueError("Expression complexity limit exceeded (Too many AST nodes).")

        if isinstance(node, ast.Expression):
            return self.evaluate(node.body, depth + 1)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                val_str = str(node.value).lower()
                if 'e' in val_str:
                    base, exponent = val_str.split('e')
                    clean_base = base.replace('.', '').replace('-', '')
                    total_implied_digits = len(clean_base) + abs(int(exponent))
                    if total_implied_digits > self.MAX_DIGITS:
                        raise ValueError(f"Numeric constant exceeds maximum permitted size of {self.MAX_DIGITS} digits.")
                else:
                    clean_str = val_str.replace('.', '').replace('-', '')
                    if len(clean_str) > self.MAX_DIGITS:
                        raise ValueError(f"Numeric constant exceeds maximum permitted size of {self.MAX_DIGITS} digits.")
            return node.value

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self.evaluate(node.operand, depth + 1)
            return self._enforce_magnitude(self.OPERATORS[op_type](operand))

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            
            left = self.evaluate(node.left, depth + 1)
            right = self.evaluate(node.right, depth + 1)

            if op_type == ast.Pow:
                if not isinstance(right, (int, float)) or abs(right) > self.MAX_POWER:
                    raise ValueError(f"Exponent power exceeds safe absolute threshold ({self.MAX_POWER}).")
                
                if isinstance(left, (int, float)) and left != 0:
                    estimated_digits = abs(float(right)) * math.log10(abs(float(left)))
                    if estimated_digits > self.MAX_RESULT_DIGITS:
                        raise ValueError("Calculation rejected: result size exceeds safe numeric memory limits.")

            try:
                result = self.OPERATORS[op_type](left, right)
                return self._enforce_magnitude(result)
            except ZeroDivisionError:
                raise ValueError("Division by zero error.")

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self.FUNCTION_SIGNATURES:
                raise ValueError("Unauthorized or unknown function call.")
            
            func_name = node.func.id
            func_impl, expected_arity = self.FUNCTION_SIGNATURES[func_name]
            
            if len(node.args) != expected_arity:
                raise ValueError(f"Arity Validation Mismatch: Function '{func_name}' expects {expected_arity} arguments.")
            
            args = [self.evaluate(arg, depth + 1) for arg in node.args]
            return self._enforce_magnitude(func_impl(*args))

        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTION_SIGNATURES:
                func_impl, expected_arity = self.FUNCTION_SIGNATURES[node.id]
                if expected_arity == 0:
                    return self._enforce_magnitude(func_impl())
            raise ValueError(f"Variable reference not permitted: {node.id}")

        raise ValueError(f"Prohibited expression element: {type(node).__name__}")

    def _enforce_magnitude(self, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Calculation rejected: result is Infinity or NaN.")
        if isinstance(value, (int, float)) and abs(value) > self.MAX_RESULT_MAGNITUDE:
            raise ValueError("Calculation rejected: resulting magnitude exceeds system bounds.")
        return value

def numpy_exec_safe(expr: str) -> str:
    try:
        if len(expr) > 500:
            return "Safety Violation: Input expression too long."
        tree = ast.parse(expr.strip(), mode='eval')
        res = ResourceConstrainedEvaluator().evaluate(tree)
        return str(float(res) if hasattr(res, '__float__') else res)
    except Exception as e:
        logger.warning(f"Math calculation engine rejected input payload: {e}")
        return f"Calculation Error: {e}"
