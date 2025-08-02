import os
import re

# Get the path to the dictionary file from an environment variable
import os

import src.config as config

def parse_custom_dict(file_path):
    """
    Parses a custom-formatted text file into a nested dictionary.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        dict: The parsed nested dictionary.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    stack = []
    current_dict = {}
    stack.append(current_dict)

    for line_number, line in enumerate(lines, 1):
        original_line = line  # Keep the original line for debugging
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue

        # Skip lines that are just closing braces
        if line == '}':
            if len(stack) > 1:
                stack.pop()
                current_dict = stack[-1]
            else:
                print(f"Warning: Unmatched closing brace at line {line_number}")
            continue

        # Check if the line starts a new dictionary
        if line.endswith('{'):
            key = line[:-1].strip()
            # Remove any trailing '=' if present
            if key.endswith('='):
                key = key[:-1].strip()
            # Remove any trailing ':' if present
            if key.endswith(':'):
                key = key[:-1].strip()
            # Assign an empty dictionary to this key
            new_dict = {}
            current_dict[key] = new_dict
            # Push the new dictionary onto the stack
            stack.append(new_dict)
            current_dict = new_dict
            continue

        # Handle key-value pairs
        if '=' in line:
            key, value = line.split('=', 1)
        elif ':' in line:
            key, value = line.split(':', 1)
        else:
            print(f"Warning: Unrecognized line format at line {line_number}: {original_line}")
            continue

        key = key.strip()
        value = value.strip()

        # Convert numeric values to integers if possible
        if value.isdigit():
            value = int(value)
        else:
            # Optionally, you can handle other data types here (e.g., dates)
            pass

        current_dict[key] = value

    return stack[0]

# Example usage
if __name__ == "__main__":
    file_path = config.TOMATOES_DICT_PATH
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        tomatoes_dict = parse_custom_dict(file_path)
        # For demonstration, we'll print the dictionary
        import pprint
        pprint.pprint(tomatoes_dict)
