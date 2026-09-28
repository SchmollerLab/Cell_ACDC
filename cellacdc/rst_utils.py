import os
import re
from qtpy.QtCore import Qt, QSortFilterProxyModel
from . import tooltips_rst_filepath

def format_bullet_points(text): #indentation for bullet points in tooltips. Implementation not robust
    lines = text.split('\n')
    formatted_lines = []
    indent = False
    indentNo = 0

    for line in lines:
        if line.strip().startswith("* "):
            indent = True
            formatted_line = line
            indentNo = len(line) - len(line.lstrip())
        else:
            indentNoComp = len(line) - len(line.lstrip())
            if indent == True and indentNo == indentNoComp:
                formatted_line = " " * 2 + line
            else:
                formatted_line = line
                indent = False

        formatted_lines.append(formatted_line)

    return '\n'.join(formatted_lines)   

def format_number_list(text): #indentation for number points in tooltips. Implementation not robust
    lines = text.split('\n')
    formatted_lines = []
    indent = False
    indentNo = 0

    for line in lines:
        if line.strip().startswith((
                "0. ", "1. ", "2. ", "3. ", "4. ", 
                "5. ", "6. ", "7. ", "8. ", "9. "
            )):
            indent = True
            formatted_line = line
            indentNo = len(line) - len(line.lstrip())
        else:
            indentNoComp = len(line) - len(line.lstrip())
            if indent == True and indentNo == indentNoComp:
                formatted_line = " " * 3 + line
            else:
                formatted_line = line
                indent = False

        formatted_lines.append(formatted_line)

    return '\n'.join(formatted_lines)

def autoLineBreak(text, length): #automatic line breaking for tooltips. Keeps indentation with spaces and preexisting line breaks
    lines = []
    current_line = []

    # Split the text into lines while preserving existing newline characters
    existing_lines = text.split('\n')

    for existing_line in existing_lines:
        # Calculate the indentation for the current line
        indent = len(existing_line) - len(existing_line.lstrip())
        words = existing_line.lstrip().split()  # Split each line into words

        for word in words:
            if len(' '.join(current_line + [word])) + indent <= length:
                current_line.append(word)
            else:
                lines.append(' ' * indent + ' '.join(current_line))
                current_line = [word]

        if current_line:  # Add any remaining words as the last line
            lines.append(' ' * indent + ' '.join(current_line))

        # Reset the current line for the next existing line
        current_line = []

    return '\n'.join(lines)

def get_tooltips_from_docs(for_search=False): 
    # gets tooltips for GUI from .\Cell_ACDC\docs\source\tooltips.rst
    var_pattern = r"\|(\S*)\|"
    shortcut_pattern = r"\*\*(\".*\")\):\*\*"
    title_pattern = r"\*\*(.*)\(\*\*"

    if not os.path.exists(tooltips_rst_filepath):
        return {}
    
    with open(tooltips_rst_filepath, "r") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if not (line.startswith("..") or line.startswith("    :target:") or line.startswith("    :alt:") or line.startswith("    :width:") or line.startswith("    :height:") or line==""):
            new_lines.append(line)
    lines = new_lines

    non_empty_lines = [line.replace("\n", "") for line in lines if line.strip()] #also removes \n from lines
    lines = non_empty_lines

    if not for_search:
        tipdict = {}
    else:
        tipdict = []

    for i, line in enumerate(lines):
        match = re.search(var_pattern, line)
        if match:
            name = match.group(1)

            title = re.search(title_pattern, line).group(1)

            shortcut = re.search(shortcut_pattern, line)
            if shortcut:
                shortcut = shortcut.group(1)
            else:
                shortcut = "\"No shortcut\""

            desc = line.split("):**")[1].lstrip(" ")

            appSameLine = False

            if desc == "":
                appSameLine = True

            descList = []
            i += 1
            for followLine in lines[i:]:
                followMatch = re.search(var_pattern, followLine)
                if followMatch or followLine.startswith("* **"):
                    break
                else:                    
                    descList.append(followLine)

            if descList != []:
                if descList[-1].startswith("----"):
                    descList.pop(-1)
                    descList.pop(-1)


            for entry in descList:

                entry = entry.replace("| ", "")

                if entry.startswith(" " * 4):
                    stripped_string = entry[4:]
                else:
                    stripped_string = entry
                entry = stripped_string

                if appSameLine == False:
                    entry = "\n" + entry
                else:
                    appSameLine = False

                desc += entry
                
            if not for_search:
                desc = autoLineBreak(desc, 60)
                desc = format_bullet_points(desc)
                desc = format_number_list(desc)

                tipdict[name] = f"Name: {title}\nShortcut: {shortcut}\n\n{desc}"
            else:
                tipdict.append({
                    'name': title,
                    'shortcut': shortcut,
                    'tooltip': desc,
                    'id': name
                })
    return tipdict

# def parse_rst_file(filepath):
#     """
#     Extract button names and tooltips from Cell-ACDC RST file.
    
#     Handles multiple formats:
#     - * **Button name (** |buttonId| **"Shortcut"):** Tooltip
#     - * **Button name (** |buttonId| **):** Tooltip (no shortcut)
#     - Multi-line tooltips with | continuation
#     """
#     buttons = []
    
#     with open(filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     # Split by bullet points (lines starting with *)
#     # Match the full entry including multi-line continuations
#     pattern = r'\*\s+\|?\s*\*\*(.+?)\s*\(\*\*\s*\|(\w+)\|\s*\*\*(?:"([^"]*)")?\)\:\*\*\s+(.+?)(?=\n\s*\*\s+\|?|\n\n[A-Za-z]|\Z)'
    
#     matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
    
#     for match in matches:
#         button_name = match.group(1).strip()
#         button_id = match.group(2).strip()
#         shortcut = match.group(3).strip() if match.group(3) else ""
#         tooltip_raw = match.group(4).strip()
        
#         # Clean up the tooltip: remove RST formatting
#         tooltip = clean_rst_text(tooltip_raw)
        
#         buttons.append({
#             'name': button_name,
#             'id': button_id,
#             'shortcut': shortcut,
#             'tooltip': tooltip
#         })
    
#     return buttons
 
 
# def clean_rst_text(text):
#     """Remove RST markup and formatting from text"""
#     # Remove RST literal blocks (.. code-block::)
#     text = re.sub(r'\.\.\s+\w+::', '', text)
    
#     # Remove image references |iconName|
#     text = re.sub(r'\|(\w+)\|', r'\1', text)
    
#     # Remove hyperlink targets (.. _target:)
#     text = re.sub(r'\.\.\s+_\w+:', '', text)
    
#     # Remove bold/italic formatting
#     text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
#     text = re.sub(r'~~(.+?)~~', r'\1', text)
    
#     # Remove inline literals (backticks)
#     text = re.sub(r'``(.+?)``', r'\1', text)
    
#     # Clean up bullet points (leading * in multi-line content)
#     text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    
#     # Clean up RST line continuation markers (| at start of line)
#     text = re.sub(r'\n\s*\|\s+', ' ', text)
    
#     # Replace multiple newlines with space (collapse multi-line entries)
#     text = re.sub(r'\n+', ' ', text)
    
#     # Clean up multiple spaces to single space
#     text = re.sub(r'\s+', ' ', text)
    
#     # Remove leading/trailing whitespace
#     text = text.strip()
    
#     return text