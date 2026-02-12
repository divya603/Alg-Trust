"""
Tokenizer for arithmetic expressions
Supports: +, -, *, /, ^ operators, negative numbers, and brackets (), {}, []
"""

import re

# All bracket types treated equivalently for expression grouping
OPEN_BRACKETS = ['(', '{', '[']
CLOSE_BRACKETS = [')', '}', ']']
BRACKET_PAIRS = {'(': ')', '{': '}', '[': ']'}  # Maps opening to closing


def tokenize(expression: str) -> list:
    """
    Tokenize an arithmetic expression into a list of tokens.
    
    Args:
        expression: String like "2+3*5" or "-3+4*2"
    
    Returns:
        List of tokens: ["2", "+", "3", "*", "5"] or ["-3", "+", "4", "*", "2"]
    
    Examples:
        >>> tokenize("2+3*5")
        ['2', '+', '3', '*', '5']
        >>> tokenize("-3+4*2")
        ['-3', '+', '4', '*', '2']
        >>> tokenize("10/2^3")
        ['10', '/', '2', '^', '3']
    """
    # Remove all whitespace
    expression = expression.replace(" ", "")
    
    tokens = []
    i = 0
    
    while i < len(expression):
        # Check if current character is a bracket (any type)
        if expression[i] in OPEN_BRACKETS + CLOSE_BRACKETS:
            tokens.append(expression[i])
            i += 1

        # Check if current character is an operator
        elif expression[i] in ['+', '*', '/', '^']:
            tokens.append(expression[i])
            i += 1

        # Handle minus: could be subtraction or negative number
        elif expression[i] == '-':
            # It's a negative number if:
            # 1. It's at the start of expression, OR
            # 2. Previous token is an operator, OR
            # 3. Previous token is an opening bracket
            if i == 0 or (tokens and (tokens[-1] in ['+', '-', '*', '/', '^'] or tokens[-1] in OPEN_BRACKETS)):
                # It's a negative number - read the full number
                j = i + 1
                while j < len(expression) and (expression[j].isdigit() or expression[j] == '.'):
                    j += 1
                tokens.append(expression[i:j])
                i = j
            else:
                # It's a subtraction operator
                tokens.append('-')
                i += 1
        
        # Handle numbers (including decimals)
        elif expression[i].isdigit() or expression[i] == '.':
            j = i
            while j < len(expression) and (expression[j].isdigit() or expression[j] == '.'):
                j += 1
            tokens.append(expression[i:j])
            i = j
        
        else:
            raise ValueError(f"Invalid character in expression: {expression[i]}")
    
    return tokens


def validate_tokens(tokens: list) -> bool:
    """
    Validate that tokens form a valid expression.
    Uses stack-based approach for bracket matching.

    Args:
        tokens: List of tokens

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not tokens:
        raise ValueError("Empty token list")

    operators = ['+', '-', '*', '/', '^']

    # Stack-based bracket matching: stores (bracket_char, position) tuples
    bracket_stack = []

    for i, token in enumerate(tokens):
        # Handle opening brackets - push onto stack
        if token in OPEN_BRACKETS:
            bracket_stack.append((token, i))

        # Handle closing brackets - pop and validate match
        elif token in CLOSE_BRACKETS:
            if not bracket_stack:
                raise ValueError(f"Unmatched closing bracket '{token}' at position {i}")

            open_bracket, open_pos = bracket_stack.pop()
            expected_close = BRACKET_PAIRS[open_bracket]

            if token != expected_close:
                raise ValueError(
                    f"Mismatched brackets: '{open_bracket}' at position {open_pos} "
                    f"closed with '{token}' at position {i} (expected '{expected_close}')"
                )

        # Check for empty brackets like (), {}, []
        if token in CLOSE_BRACKETS and i > 0 and tokens[i-1] in OPEN_BRACKETS:
            raise ValueError(f"Empty brackets at position {i}")

        # Check that operators have valid neighbors
        if token in operators:
            # Operator shouldn't be at start (except minus handled as negative)
            if i == 0:
                raise ValueError(f"Expression cannot start with operator: {token}")
            # Operator shouldn't be at end
            if i == len(tokens) - 1:
                raise ValueError(f"Expression cannot end with operator: {token}")
            # Previous token should be a number or closing bracket
            prev = tokens[i-1]
            if prev in operators or prev in OPEN_BRACKETS:
                raise ValueError(f"Operator {token} at position {i} follows invalid token: {prev}")
            # Next token should be a number or opening bracket
            next_token = tokens[i+1]
            if next_token in operators or next_token in CLOSE_BRACKETS:
                raise ValueError(f"Operator {token} at position {i} precedes invalid token: {next_token}")

    # Check for unclosed brackets
    if bracket_stack:
        unclosed = ', '.join(f"'{b}' at position {p}" for b, p in bracket_stack)
        raise ValueError(f"Unclosed brackets: {unclosed}")

    return True


if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2+3*5",
        "-3+4*2",
        "10/2^3",
        "5-3*2",
        "2*-3",
        "100+50-25*2/5",
        "(2+3)*5",
        "2*(3+4)",
        "((2+3))",
        "(2+3)*(4+5)",
        "(-3+4)*2",
        # New bracket types
        "{2+3}*5",
        "[2+3]*5",
        "{[2+3]}*5",
        "({2+3})*[4+5]",
        # Nested brackets
        "((2+3))*5",
        "{(2+3)}*5",
        "[{(2+3)}]*5",
    ]

    print("Testing tokenizer:")
    print("-" * 50)
    for expr in test_expressions:
        tokens = tokenize(expr)
        validate_tokens(tokens)
        print(f"{expr:25} -> {tokens}")

    # Test invalid cases
    print("\n" + "-" * 50)
    print("Testing invalid expressions (should raise errors):")
    print("-" * 50)
    invalid_expressions = [
        ("(2+3]", "Mismatched brackets"),
        ("{2+3)", "Mismatched brackets"),
        ("(2+3", "Unclosed bracket"),
        ("2+3)", "Unmatched closing"),
        ("()", "Empty brackets"),
        ("{}", "Empty brackets"),
    ]
    for expr, expected_error in invalid_expressions:
        try:
            tokens = tokenize(expr)
            validate_tokens(tokens)
            print(f"{expr:15} -> ERROR: Should have raised exception")
        except ValueError as e:
            print(f"{expr:15} -> OK: {e}")
